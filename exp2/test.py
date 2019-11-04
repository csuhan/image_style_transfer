import argparse
import random

import cv2
import numpy as np
import torch
from torchvision import datasets, transforms

from base_model import TransformNet
from utils import debug_result, read_image, recover_image


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',type=str)
    parser.add_argument('content_img', type=str)
    parser.add_argument('-o', '--out', default='out.jpg')
    return parser.parse_args()


args = arg_parser()
model_path = args.model
content_img_path = args.content_img
style_img_path = '../imgs/style.jpg'

style_img = cv2.imread(style_img_path)
h,w = style_img.shape[:2]
style_img = cv2.resize(style_img,(int(w*500/h),500))
content_img = cv2.imread(content_img_path)
input_img = read_image(content_img_path)

trans_net = TransformNet(16)
trans_net.load_state_dict(torch.load(model_path))

out = trans_net(input_img)
out = recover_image(out)
cv2.imwrite(args.out, out)
res = debug_result(style_img, content_img, out)
res.save(args.out.split('.')[0]+'_debug.jpg')
