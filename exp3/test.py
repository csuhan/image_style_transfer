import os
import random

import cv2
import numpy as np
import torch
from PIL import Image

from base_model import VGG, MetaNet, TransformNet
from utils import (debug_result, get_dataset, get_transform, mean_std,
                   read_image, recover_image)


torch.cuda.set_device(1)
vgg = VGG().cuda().eval()
transform_net = TransformNet(8).cuda().eval()
meta_net = MetaNet(transform_net.get_param_dict()).cuda().eval()

transform_net.load_state_dict(torch.load('models/transformnet.pth'))
meta_net.load_state_dict(torch.load('models/metanet.pth'))

style_image = read_image('../imgs/style.jpg').cuda()
content_image = read_image('../imgs/content1.jpg').cuda()
style_feature = vgg(style_image)

weights = meta_net(mean_std(style_feature))

transform_net.set_weights(weights, 0)

out = transform_net(content_image)
out = recover_image(out)
res = debug_result(recover_image(style_image),
                   recover_image(content_image), out)
res.save('out_debug.jpg')
