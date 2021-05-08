from .lib.MNet import MNet
import torch
import os
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
        transforms.Resize((299, 299)),  # TODO: resize for different input image size
        transforms.ToTensor()]
)
def loader(path):
    return Image.open(path).convert('RGB')
img = loader("816eb92432f270c308c4f7da4636d7036e55b244.jpg")
img = transform(img).unsqueeze(0)
net = MNet()
net.load_state_dict(torch.load('checkpoint\MNet_epoch_995'))

import scipy.io as scio
 
dataFile = './annotation.mat'
data = scio.loadmat(dataFile)
attrs = [i.item()[0] for i in data['attributes']]
print(attrs)
torch.sigmoid(net(img))[0]
