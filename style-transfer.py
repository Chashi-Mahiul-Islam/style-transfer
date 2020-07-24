# style transfer paper:
# https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf


from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

import torch 
from torch import optim
import requests
from torchvision import transforms, models

 # load feature portion of VGG19
vgg = models.vgg19(pretrained=True).features
 
 # we are only optimizing the target image so we don't need to change any parameter
 # so we freeze the parameters
for param in vgg.parameters():
     param.requires_grad_(False)
  
# move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg.to(device)

def load_image(img_path, max_size=400, shape=None):
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')
    
    # large images will sloww down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
        
    if shape is not None:
        size = shape
    
    in_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
    
    # discard transparent  alpha channel ( :3) and add the batch dimension
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    
    return image

# load content and style images
content = load_image('images/space_needle.jpg').to(device)
style = load_image('images/hockney.jpg', shape=content.shape[-2:]).to(device) 
# shape -> 0th index how many image, 1th index no of channels, 2nd index height, 3rd index width
# [-2:] means height and width

# helper function for un-normalizing an image 
# and converting it from a Tensor image to a NumPy image for display
def im_convert(tensor):
    """ Display a tensor as an image. """
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

# display the images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# content and style ims side-by-side
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(style))

# VGG has 19 layers
# here for content feature we use 4_2 conv layer
# and for style features we use 1_1, 2_1, 3_1, 4_1 and 5_1 conv layers

# print(vgg)

# content and style features
def get_features(image, model, layers=None):
    # run an image forward through vgg and get features from various conv layers
    
    # naming layers according to paper
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2', ## content representation
                  '28': 'conv5_1'}
        
    features = {}
    x = image
    
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x # for name=0 features['conv1_1'] = conv1_1(x)
            # in features list every item is a conv layer with various depth 
    return features


# style representation is compared through gram matrix 
# output1 = conv1_1(x) - it is a conv layer from conv1_1
# here it has d depth, h height, w width 
# we convert a matrix, m = (d, h*d)
# gram_matrix = m*mT 
    
def gram_matrix(tensor):
    
    # get the batch_size, depth, height, and width of the tensor
    _, d, h, w = tensor.size()
    
    # get the m matrix where shape = (d, h*w)
    tensor = tensor.view(d, h*w)
    
    # get the gram matrix m * mTranspose
    gram = torch.mm(tensor, tensor.t())
    
    return gram

# features and style_grams
# get initial content and style features before training
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# gram matrices for each layer of our style representation
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
# {'conv1_1': m1*m1T, 'conv2_1': m2*m2T...}

# we need to create a taget image and gradually change it through training
# good idea: content image as target for starting
# then iteratively change its style

#target = content.clone().requires_grad_(True).to(device)

#target = torch.Tensor(content.shape).requires_grad_(True).to(device)
target = torch.rand(content.shape).to(device).requires_grad_(True).to(device)

# loss and weight
# weighting earlier layers: expect larger style artifacts
# weighting later layers: emphasis on smaller features
# excluding 'conv4_2' as it was for content representation
style_weights = {'conv1_1': 1.0,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

content_weight = 1 # alpha
style_weight = 1e3 # beta

# content_loss = 1/2*sum(Cc-Tc)^2
# style_loss = a*sum(wi*(Tsi -Ssi)^2)
# total_loss = alpha*content_loss + beta*style_loss

# train

# changes image showing steps
show_every = 200

optimizer = optim.Adam([target], lr = 0.003)
steps = 2000 

for i in range(steps):
    if i%40 == 0:
        print(i+1)
    # get feature from my target image
    target_features = get_features(target, vgg)
    
    # the content loss 
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    
    # the style loss
    style_loss =0
    
    for layer in style_weights:
        # get 'target' style representation for the layer
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        # get the 'style' style representation 
        style_gram = style_grams[layer]
        
        # the style loss for one layer, weighted 
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        
        # add to total style_loss from all layer
        style_loss += layer_style_loss / (d*h*w)
        
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    # update target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if i % show_every == 0:
        print('Total loss: ', total_loss.item())
        plt.imshow(im_convert(target))
        plt.show()

# display content and final, target image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(target))



