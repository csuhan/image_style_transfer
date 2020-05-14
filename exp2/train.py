import argparse

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

from base_model import VGG, TransformNet
from utils import get_data_loader, get_dist_data_loader, gram_matrix, read_image,save_debug_image

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('coco_path',type=str)
    parser.add_argument('style_image',type=str)
    parser.add_argument('--local_rank',type=int, default=0)
    parser.add_argument('-n','--model_name',default='transform_net')
    return parser.parse_args()

# coco_path = '/data/hjm/coco/'
# style_image_path = '../imgs/style.jpg'
args = arg_parser()
coco_path = args.coco_path
style_image_path = args.style_image
model_name = args.model_name
width = 256
batch_size = 32
verbose_batch = 50
style_weight = 1e5
content_weight = 1
tv_weight = 1e-6
lr = 1e-3
net_base = 32
epoches = 5

# data preprocess
data_loader = get_data_loader(coco_path, batch_size, width)

# vgg and style image
vgg16 = VGG().cuda().eval()

style_img = read_image(style_image_path).cuda()
style_features = vgg16(style_img)
style_grams = [gram_matrix(x).detach() for x in style_features]

transform_net = TransformNet(net_base).cuda()
optimizer = optim.Adam(transform_net.parameters(), lr)
transform_net.train()

n_batch = len(data_loader)
for epoch in range(epoches):
    print('Epoch:{}'.format(epoch+1))

    with tqdm(enumerate(data_loader), total=n_batch) as pbar:
        for batch, (content_images, _) in pbar:
            optimizer.zero_grad()
            # input image -> transform_net -> transformed images
            content_images = content_images.cuda()
            content_features = vgg16(content_images)

            transformed_images = transform_net(content_images)
            transformed_images = transformed_images.clamp(-3, 3)
            transformed_features = vgg16(transformed_images)

            content_loss = content_weight * \
                F.mse_loss(transformed_features[1], content_features[1])

            # tv loss
            y = transformed_images
            tv_loss = tv_weight*(torch.sum(torch.abs(y[:, :, :, :-1]-y[:, :, :, 1:]))+torch.sum(
                torch.abs(y[:, :, :-1, :]-y[:, :, 1:, :])))

            style_loss = 0
            transformed_grams = [gram_matrix(x) for x in transformed_features]
            for transformed_gram, style_gram in zip(transformed_grams, style_grams):
                style_loss += style_weight * F.mse_loss(transformed_gram,
                                                        style_gram.expand_as(transformed_gram))

            loss = style_loss + content_loss + tv_loss

            loss.backward()
            optimizer.step()

            s = "Content Loss {:.4f} Style Loss {:.4f} Tv Loss {:.4f} Loss {:.4f}"\
                .format(content_loss.item(), style_loss.item(), tv_loss.item(), loss.item())
            pbar.set_description(s)

            if batch % verbose_batch == 0:
                save_debug_image(style_img,content_images,transformed_images,width,'debug_images/{}.jpg'.format(batch))

    torch.save(transform_net.state_dict(), 'models/{}.pth'.format(model_name))
