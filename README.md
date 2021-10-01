# Style Transfer : Using Convolutional Neural Network 

#### Project Status: [Active, On-Hold, Completed]
Completed

## Project Intro/Objective
This repo contains the code base for transfering one image's style to another image. This is an implementation of a paper named "Image Style Transfer Using Convolutional Neural Networks". Paper link: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf

## Project Description
 I used a pretrained VGG 19 network.  I extracted the content and style representationsof images from unique layers.  The aim was to create a new target image that tries tobalance the content of one image with the style of another. I usedPyTorchfor buildingthe whole architecture.

## Installation   
 * git clone https://github.com/Chashi-Mahiul-Islam/style-transfer.git
 * Run "poetry install" from root directory to install all required libraries
 * Run "poetry shell" to activate the virtual env
 * To run the project follow the Usage
 
## Usage
```
    Style Transfer CL
    
    Usage: driver.py [OPTIONS]
    python3 driver.py --source=<image-path> --target=<image-path --steps=<steps-to-train> --show-every=<show-intermediary-images> --learning-rate=<learning-rate>
    Options:
      --source TEXT          Source image to copy the style
      --target TEXT          Target image to apply the style
      --steps INTEGER        Total steps to train
      --show_every INTEGER   When to show the current state of the output image
      --learning_rate FLOAT  Learning rate
      --help                 Show this message and exit.


```

## Example

``` python3 driver.py --source="style_transfer/images/hockney.jpg" --target="style_transfer/images/octopus.jpg"  ``` 


### Technologies
* Python
* PyTorch




