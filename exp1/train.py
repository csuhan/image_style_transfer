import argparse

import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from base_model import VGG
from utils import gram_matrix, load_image, recover_image


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('style', type=str)
    parser.add_argument('-o', '--out', type=str, default='out.jpg')
    parser.add_argument('-s', '--steps', type=int, default=300)
    return parser.parse_args()


# 使用gpu:0
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = arg_parser()

style_weight = 1e7
content_weight = 1
train_steps = args.steps

content_img = load_image(args.input).to(device, torch.float)
style_img = load_image(args.style).to(device, torch.float)
input_img = content_img.clone()

vgg16 = VGG().cuda().eval()
optimizer = optim.LBFGS([input_img.requires_grad_()])

content_features = vgg16(content_img)
style_features = vgg16(style_img)
style_grams = [gram_matrix(x) for x in style_features]

step = [0]
while step[0] <= train_steps:
    def f():
        optimizer.zero_grad()
        features = vgg16(input_img)

        content_loss = F.mse_loss(
            features[2], content_features[2])*content_weight
        style_loss = 0
        grams = [gram_matrix(x) for x in features]
        for a, b in zip(grams, style_grams):
            style_loss += F.mse_loss(a, b)*style_weight

        loss = style_loss+content_loss

        if step[0] % 50 == 0:
            print('Step {}: Style Loss: {:4f} Content Loss: {:4f} Total Loss: {:4f}'.format(
                step[0], style_loss.item(), content_loss.item(), loss.item()
            ))
        step[0] += 1

        loss.backward()
        return loss

    optimizer.step(f)


out = recover_image(input_img)
cv2.imwrite(args.out, out)
