import pickle
import numpy as np
import yaml
import torch
from mmcv.runner import load_checkpoint
from mmskeleton.models.backbones.st_gcn_aaai18 import ST_GCN_18
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

'''
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
'''
cur_path = os.path.dirname(os.path.realpath(__file__))+'/'
labels_file = cur_path + \
    'deprecated/origin_stgcn_repo/resource/kinetics_skeleton/label_name.txt'
cfg_file = cur_path+'configs/recognition/st_gcn_aaai18/kinetics-skeleton/test.yaml'
def get_labels():
    labels = []
    with open(labels_file, 'r') as f:
        for line in f:
            labels.append(line.strip('\n'))
    return labels

def get_model():
    cfg = yaml.load(open(cfg_file, "rb"), Loader=yaml.CLoader)
    cfg = cfg['processor_cfg']['model_cfg']
    cfg.pop('type')
    model = ST_GCN_18(**cfg)
    model.eval()
    load_checkpoint(model, cur_path+'checkpoints/st_gcn.kinetics-6fa43f73.pth')
    return model





def pose_normal(pose, W=192, H=256):
    pose[:, 0] /= W
    pose[:, 1] /= H
    pose[:, 0:2] = pose[:, 0:2] - 0.5
    pose[:, 0][pose[:, 2] == 0] = 0
    pose[:, 0][pose[:, 2] == 0] = 0
    return pose

def inference(model,pose):
    '''
    input pose tensor shape should like (lengths of):
    
    [batch,values[x,y,score],time_steps,joints(graph_nodes),human_objects]  
    
    output shapes like, 
     
    if multi human in image,it only analysis their action in whole !  
    
    not logit,call softmax yourself:  
    
    [batch,classes]

    '''
    return model(pose)
if __name__ == '__main__':
    pose = torch.tensor(pickle.load(open('pose.pkl', 'rb'))[0]['keypoints'])
    pose = pose_normal(pose).transpose(-1, -2).view(1, 3, 1, 18, 1)
    model = get_model()
    print(pose.shape)
    o = model(pose)
    print(labels[torch.argmax(o).item()], o.shape)
