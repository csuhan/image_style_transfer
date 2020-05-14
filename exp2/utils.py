import torch
import cv2
from PIL import Image
import numpy as np
from torchvision import datasets, transforms

cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]
tensor_normalizer = transforms.Normalize(
    mean=cnn_normalization_mean, std=cnn_normalization_std)
'''
计算Gram矩阵，用于评估Style Loss
'''


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w*h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)/(ch*h*w)
    return gram


def get_data_loader(path, batch_size=4, img_width=256):
    data_transform = transforms.Compose([
        transforms.Resize(img_width),
        transforms.CenterCrop(img_width),
        transforms.ToTensor(),
        tensor_normalizer,
    ])
    dataset = datasets.ImageFolder(path, transform=data_transform)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def get_dist_data_loader(path, batch_size=4, img_width=256, sampler=None):
    data_transform = transforms.Compose([
        transforms.Resize(img_width),
        transforms.CenterCrop(img_width),
        transforms.ToTensor(),
        tensor_normalizer,
    ])
    dataset = datasets.ImageFolder(path, transform=data_transform)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler(dataset))
    return data_loader



def read_image(path):
    '''
    read image from path
    return 4-d tensor
    '''
    t = transforms.Compose([
        transforms.ToTensor(),
        tensor_normalizer
    ])
    img = cv2.imread(path)
    return t(img).unsqueeze(0)


def recover_image(tensor):
    """输入 GPU 上的四维 tensor，输出 0~255 范围的三维 numpy 矩阵，RGB 顺序"""
    image = tensor.detach().cpu().numpy()
    image = image * np.array(cnn_normalization_std).reshape((1, 3, 1, 1)) + \
        np.array(cnn_normalization_mean).reshape((1, 3, 1, 1))
    return (image.transpose(0, 2, 3, 1) * 255.).clip(0, 255).astype(np.uint8)[0]


def debug_result(style_img, content_img, out_img):
    h0, w0 = style_img.shape[:2]
    h1, w1 = content_img.shape[:2]
    h2, w2 = out_img.shape[:2]
    height = max(h0, h1, h2)
    width = w0+w1+w2

    def t(im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return Image.fromarray(np.uint8(im))
    style_img, content_img, out_img = t(style_img), t(content_img), t(out_img)

    im = Image.new('RGB', (width, height))
    im.paste(style_img, (0, 0))
    im.paste(content_img, (w0, 0))
    im.paste(out_img, (w0+w1, 0))

    return im

def save_debug_image(style_images, content_images, transformed_images, width,filename):
    style_image = Image.fromarray(recover_image(style_images))
    w1,h1 = style_image.width*(width*2+5)/style_image.height,width*2+5
    style_image = style_image.resize((int(w1),h1))
    content_images = [cv2.cvtColor(recover_image(x),cv2.COLOR_BGR2RGB) for x in content_images]
    transformed_images = [cv2.cvtColor(recover_image(x),cv2.COLOR_BGR2RGB) for x in transformed_images]
    
    new_im = Image.new('RGB', (style_image.size[0] + (width + 5) * 4, max(style_image.size[1], width*2 + 5)))
    new_im.paste(style_image, (0,0))
    
    x = style_image.size[0] + 5
    for i, (a, b) in enumerate(zip(content_images, transformed_images)):
        new_im.paste(Image.fromarray(a), (x + (width + 5) * i, 0))
        new_im.paste(Image.fromarray(b), (x + (width + 5) * i, width + 5))
    
    new_im.save(filename)
