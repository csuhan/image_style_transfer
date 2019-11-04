import cv2
import numpy as np
import torch


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w*h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)/(ch*h*w)
    return gram


def load_image(file_path):
    img = cv2.imread(file_path)
    img = np.transpose(img, axes=[2, 0, 1])
    img = img/255
    img = torch.Tensor(img).unsqueeze(0)
    return img


def recover_image(img):
    out = img.detach().cpu().numpy().squeeze(0).transpose([1, 2, 0])
    out = out*255
    return out