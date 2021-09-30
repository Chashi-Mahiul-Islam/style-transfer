from torchvision import models

def load_vgg(verbose=False):
    # VGG has 19 layers
    # here for content feature we use 4_2 conv layer
    # and for style features we use 1_1, 2_1, 3_1, 4_1 and 5_1 conv layers

    # load feature portion of VGG19
    vgg = models.vgg19(pretrained=True).features
     
     # we are only optimizing the target image so we don't need to change any parameter
     # so we freeze the parameters
    for param in vgg.parameters():
         param.requires_grad_(False)
    if verbose:
        print(vgg)
    return vgg