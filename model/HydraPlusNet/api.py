import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from lib.MNet import MNet
import torch
import os
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
        transforms.Resize((299, 299)), 
        transforms.ToTensor()]
)
def loader(path):
    return Image.open(path).convert('RGB')
# img = loader("816eb92432f270c308c4f7da4636d7036e55b244.jpg")
# img = transform(img).unsqueeze(0)


import scipy.io as scio

dataFile = os.path.dirname(os.path.realpath(__file__))+'/annotation.mat'
data = scio.loadmat(dataFile)
attrs = [i.item()[0] for i in data['attributes']]
class model(torch.nn.Module):
    def __init__(self,weights):
        super().__init__()
        self.net = MNet()
        self.net.load_state_dict(torch.load(weights))
        
    def forward(self,x):
        return torch.sigmoid(self.net(x))

