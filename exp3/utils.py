import torch
from PIL import Image
import numpy as np
from torchvision import datasets, transforms

cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]
tensor_normalizer = transforms.Normalize(
    mean=cnn_normalization_mean, std=cnn_normalization_std)


def get_transform(img_width):
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            img_width, scale=(256/480, 1), ratio=(1, 1)),
        transforms.CenterCrop(img_width),
        transforms.ToTensor(),
        tensor_normalizer,
    ])
    return data_transform


def get_dataset(path, img_width):
    dataset = datasets.ImageFolder(path, transform=get_transform(img_width))
    return dataset


def get_data_loader(path, batch_size=8, img_width=256):
    dataset = datasets.ImageFolder(path, transform=get_transform(img_width))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)
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
    img = Image.open(path).convert('RGB')
    return t(img).unsqueeze(0)


def recover_image(tensor):
    """输入 GPU 上的四维 tensor，输出 0~255 范围的三维 numpy 矩阵，RGB 顺序"""
    image = tensor.detach().cpu().numpy()
    image = image * np.array(cnn_normalization_std).reshape((1, 3, 1, 1)) + \
        np.array(cnn_normalization_mean).reshape((1, 3, 1, 1))
    return (image.transpose(0, 2, 3, 1) * 255.).clip(0, 255).astype(np.uint8)[0]


def mean_std(features):
    """输入 VGG16 计算的四个特征，输出每张特征图的均值和标准差，长度为1920"""
    mean_std_features = []
    for x in features:
        x = x.view(*x.shape[:2], -1)
        x = torch.cat([x.mean(-1), torch.sqrt(x.var(-1) + 1e-6)], dim=-1)
        n = x.shape[0]
        # 【mean, ..., std, ...] to [mean, std, ...]
        x2 = x.view(n, 2, -1).transpose(2, 1).contiguous().view(n, -1)
        mean_std_features.append(x2)
    mean_std_features = torch.cat(mean_std_features, dim=-1)
    return mean_std_features


def debug_result(style_img, content_img, out_img):
    h0, w0 = style_img.shape[:2]
    h1, w1 = content_img.shape[:2]
    h2, w2 = out_img.shape[:2]
    height = max(h0, h1, h2)
    width = w0+w1+w2

    def t(x): return Image.fromarray(np.uint8(x))
    style_img, content_img, out_img = t(style_img), t(content_img), t(out_img)

    im = Image.new('RGB', (width, height))
    im.paste(style_img, (0, 0))
    im.paste(content_img, (w0, 0))
    im.paste(out_img, (w0+w1, 0))

    return im

def save_debug_image(style_images, content_images, transformed_images, filename):
    width = 256
    style_image = Image.fromarray(recover_image(style_images))
    content_images = [recover_image(x) for x in content_images]
    transformed_images = [recover_image(x) for x in transformed_images]
    
    new_im = Image.new('RGB', (style_image.size[0] + (width + 5) * 4, max(style_image.size[1], width*2 + 5)))
    new_im.paste(style_image, (0,0))
    
    x = style_image.size[0] + 5
    for i, (a, b) in enumerate(zip(content_images, transformed_images)):
        new_im.paste(Image.fromarray(a), (x + (width + 5) * i, 0))
        new_im.paste(Image.fromarray(b), (x + (width + 5) * i, width + 5))
    
    new_im.save(filename)
