import os
import argparse
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm

from base_model import VGG, MetaNet, TransformNet
from utils import get_data_loader, get_dataset, mean_std,save_debug_image


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('coco_path',type=str)
    parser.add_argument('style_path',type=str)
    return parser.parse_args()

torch.cuda.set_device(0)

args = arg_parser()
net_base = 8
img_width = 256
# style_path = '/data/hjm/wikiart/'
# coco_path = '/data/hjm/coco/'
style_path = args.style_path
coco_path = args.coco_path
epoches = 7
batch_size = 8
content_weight = 1
style_weight = 25
tv_weight = 1e-6


vgg = VGG().cuda().eval()
transform_net = TransformNet(net_base).cuda().eval()
meta_net = MetaNet(transform_net.get_param_dict()).cuda().eval()

#加载预训练模型
if os.path.exists('transformnet.pth') and os.path.exists('metanet.pth'):
    transform_net.load_state_dict(torch.load('transformnet.pth'))
    meta_net.load_state_dict(torch.load('metanet.pth'))

style_dataset = get_dataset(style_path, img_width)
content_data_loader = get_data_loader(coco_path, batch_size, img_width)

train_params = {}
train_param_shapes = {}
for model in [vgg, transform_net, meta_net]:
    for name, param in model.named_parameters():
        if param.requires_grad:
            train_params[name] = param
            train_param_shapes[name] = param.shape

optimizer = optim.Adam(train_params.values(), 1e-3)

n_batch = len(content_data_loader)
meta_net.train()
transform_net.train()

style_image = random.choice(style_dataset)[0].unsqueeze(0).cuda()
style_features = vgg(style_image)
style_mean_std = mean_std(style_features)

for epoch in range(epoches):
    with tqdm(enumerate(content_data_loader), total=n_batch) as pbar:
        for batch, (content_images, _) in pbar:
            if batch % 20 == 0:
                style_image = random.choice(style_dataset)[
                    0].unsqueeze(0).cuda()
                style_features = vgg(style_image)
                style_mean_std = mean_std(style_features)

            # 去除纯色影像
            x = content_images.cpu().numpy()
            if (x.min(-1).min(-1) == x.max(-1).max(-1)).any():
                continue

            optimizer.zero_grad()

            weights = meta_net(style_mean_std)
            transform_net.set_weights(weights, 0)

            content_images = content_images.cuda()
            transformed_images = transform_net(content_images)

            content_features = vgg(content_images)
            transformed_features = vgg(transformed_images)
            transformed_mean_std = mean_std(transformed_features)

            content_loss = content_weight * \
                F.mse_loss(transformed_features[2], content_features[2])
            style_loss = style_weight * \
                F.mse_loss(transformed_mean_std,
                           style_mean_std.expand_as(transformed_mean_std))

            y = transformed_images
            tv_loss = tv_weight*(torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                                 torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

            loss = content_loss+style_loss+tv_loss

            loss.backward()
            optimizer.step()

            s = 'Epoch {} Content Loss {:.4f} Style Loss {:.4f} Tv Loss {:.4f} Total Loss {:.4f}'\
                .format(epoch, content_loss.item(), style_loss.item(), tv_loss.item(), loss.item())

            pbar.set_description(s)
            if batch % 100 == 0:
                torch.save(meta_net.state_dict(), 'metanet.pth')
                torch.save(transform_net.state_dict(), 'transformnet.pth')
                save_debug_image(style_image,content_images,transformed_images,'debug_image/{}_{}.jpg'.format(epoch,batch))

torch.save(meta_net.state_dict(), 'metanet.pth')
torch.save(transform_net.state_dict(), 'transformnet.pth')
