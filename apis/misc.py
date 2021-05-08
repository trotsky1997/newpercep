import pathlib
import os
import torch
import sys
import pickle

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
    class mixer():
        def __init__(self):
            self.model = DeepMAR_ResNet50()
            self.model.load_state_dict(torch.load(root_dir+'weights/deepmar/deepmar_statedict.pth'))
            self.model.eval()
            self.attrs = pickle.load(open(root_dir+'model/deepmar/attribute_list_Chinese.pkl'))
    return mixer()

def get_mnnet():
    from model.HydraPlusNet.api import transform,model,attrs
    class mixer():
        def __init__(self):
            self.model = model(root_dir+"weights/mnnet/MNet_epoch_995")#输入rgb
            self.transform = transform
            self.attrs = attrs
    return mixer()



def get_face()