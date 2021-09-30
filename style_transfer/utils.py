import numpy as np
import torch
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