import os
import random

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from base_model import VGG, MetaNet, TransformNet
from utils import (debug_result, get_dataset, get_transform, mean_std,
                   read_image, recover_image)

torch.cuda.set_device(1)
vgg = VGG().cuda().eval()
transform_net = TransformNet(32).cuda().eval()
meta_net = MetaNet(transform_net.get_param_dict()).cuda().eval()

transform_net.load_state_dict(torch.load('transformnet.pth'))
meta_net.load_state_dict(torch.load('metanet.pth'))

style_dataset = get_dataset('../imgs/art_dataset/', 500)
content_image = read_image('../imgs/content_meta.jpg').cuda()
for i, (style_image, _) in tqdm(enumerate(style_dataset)):
    style_image = style_image.unsqueeze(0).cuda()
    style_feature = vgg(style_image)

    weights = meta_net(mean_std(style_feature))
    transform_net.set_weights(weights, 0)

    out = transform_net(content_image)
    out = recover_image(out)
    res = debug_result(recover_image(style_image),
                       recover_image(content_image), out)
    res.save('debug_image/out_{}.jpg'.format(i))
