import pathlib
import os
import torch
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/'
sys.path.append(root_dir)






def get_yolov5():
    torch.hub.set_dir(root_dir+'weights/')
    from model.yolov5.models.yolo import Model, attempt_load
    return attempt_load(root_dir+'weights/yolo/yolov5s.pt').autoshape()

def get_sort(deep=True):
    if deep:
        from model.deep_sort_pytorch.deep_sort import DeepSort
        return DeepSort(root_dir+"weights/deepsort/ckpt.t7")
    else:
        from model.sort import Sort
        return Sort()
    
def get_deepmar():
    from model.deepmar.baseline.model.DeepMAR import DeepMAR_ResNet50
    model = DeepMAR_ResNet50()
    model.load_state_dict(torch.load(root_dir+'weights/deepmar/deepmar_statedict.pth'))
    model.eval()
    return model

def get_mnnet():
     