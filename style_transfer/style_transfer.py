# style transfer paper:
# https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
import matplotlib.pyplot as plt

import torch 
from torch import optim
import style_transfer.model_loader as ml
import style_transfer.utils as ut
import style_transfer.data_loader as dl
def transfer_style(source_path, target_path, steps=2000, show_every=200, learning_rate=0.003):
    
    # check if cuda is available or not
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # move model to GPU if available
    vgg = ml.load_vgg()
    vgg.to(device)
    
    # load content and style images
    content = dl.load_image(source_path).to(device)
    style = dl.load_image(target_path, shape=content.shape[-2:]).to(device) 
    # shape -> 0th index how many image, 1th index no of channels, 2nd index height, 3rd index width
    # [-2:] means height and width
    
    
    
    # display the images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # content and style ims side-by-side
    ax1.imshow(ut.im_convert(content))
    ax2.imshow(ut.im_convert(style))
    
    
    # features and style_grams
    # get initial content and style features before training
    content_features = ut.get_features(content, vgg)
    style_features = ut.get_features(style, vgg)
    
    # gram matrices for each layer of our style representation
    style_grams = {layer: ut.gram_matrix(style_features[layer]) for layer in style_features}
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
            print('Step: {}'.format(i))
        # get feature from my target image
        target_features = ut.get_features(target, vgg)
        
        # the content loss 
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        
        # the style loss
        style_loss =0
        
        for layer in style_weights:
            # get 'target' style representation for the layer
            target_feature = target_features[layer]
            target_gram = ut.gram_matrix(target_feature)
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
            plt.imshow(ut.im_convert(target))
            plt.show()
    
    # display content and final, target image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(ut.im_convert(content))
    ax2.imshow(ut.im_convert(target))
    
    
if __name__ == "__main__":
    transfer_style("", "")
