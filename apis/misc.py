import pathlib
import os
import torch
import sys
import pickle
import numpy as np
from torchvision.transforms.functional import pad
import cv2

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/'
sys.path.append(root_dir)
sys.path.append(root_dir+'model/yolov5/')

'''
First of all,Thanks for all the opnesource Models and their distinguished Researchers !
If You have any questions,or remind of any conduct of mine to your works,please contact me.And i'm sorry for any possible unpleasantness.

Notice that "api.py" in models/* is my rewriten scripts,If you wanna change their function,
you can change these 'api.py'.I have reserve some useful interface in that files for you to do second stage developments.

this scripts is a collection of wetworks.
'''

class mixer():
    def __init__(self,*args):
        pass
    def preprocess(self,*args):
        pass
    def postprocess(self,*args):
        pass

def get_yolov5():
    class yolomixer(mixer):
        def __init__(self):
            super(yolomixer, self).__init__()
            import torchvision.transforms as transforms
            from model.yolov5.models.yolo import Model, attempt_load
            model:Model = attempt_load(root_dir+'weights/yolo/yolov5s.pt').autoshape().eval()
            self.model = model

        def inference(self,img,aug=True):
            result = self.model(img,aug)[0]
            return result
        
        def preprocess(self,x):
            '''
            x:[c,h,w]
            out:
            x,original_size,new_size
            '''
            ori_sz = x.shape[:-1]
            x = self.letterbox(x)[0]
            x = x[:,:,[2,1,0]].transpose(2,0,1)
            rgb_img = x = torch.from_numpy(x).float()
            x /= 255.0
            if x.ndimension() == 3:
                x = x.unsqueeze(0)
            return x,ori_sz,x.shape[-2:]
        
        def postprocess(self,yolo_result,img_size,img0_size,eps=0.25):
            from model.yolov5.utils.general import non_max_suppression
            yolo_result = non_max_suppression(yolo_result,eps,classes=0)[0][:,:-1]
            # yolo_result[(yolo_result[:,5] == 0)&(yolo_result[:,4] >= eps)][:,:5]
            yolo_result[:,:4] = self.scale_coords(img_size, yolo_result[:, :4], img0_size).round()
            return yolo_result

        def scale_coords(self,img1_shape, coords, img0_shape, ratio_pad=None):
            # Rescale coords (xyxy) from img1_shape to img0_shape
            if ratio_pad is None:  # calculate from img0_shape
                gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
                pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
            else:
                gain = ratio_pad[0][0]
                pad = ratio_pad[1]

            coords[:, [0, 2]] -= pad[0]  # x padding
            coords[:, [1, 3]] -= pad[1]  # y padding
            coords[:, :4] /= gain
            self.clip_coords(coords, img0_shape)
            return coords
        
        def clip_coords(self,boxes, img_shape):
            # Clip bounding xyxy bounding boxes to image shape (height, width)
            boxes[:, 0].clamp_(0, img_shape[1])  # x1
            boxes[:, 1].clamp_(0, img_shape[0])  # y1
            boxes[:, 2].clamp_(0, img_shape[1])  # x2
            boxes[:, 3].clamp_(0, img_shape[0])  # y2
            
        def letterbox(self,img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
            # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
            shape = img.shape[:2]  # current shape [height, width]
            if isinstance(new_shape, int):
                new_shape = (new_shape, new_shape)

            # Scale ratio (new / old)
            r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
            if not scaleup:  # only scale down, do not scale up (for better test mAP)
                r = min(r, 1.0)

            # Compute padding
            ratio = r, r  # width, height ratios
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
            if auto:  # minimum rectangle
                dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
            elif scaleFill:  # stretch
                dw, dh = 0.0, 0.0
                new_unpad = new_shape
                ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

            dw /= 2  # divide padding into 2 sides
            dh /= 2

            if shape[::-1] != new_unpad:  # resize
                img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
            return img, ratio, (dw, dh)


    return yolomixer()

def get_sort(deep=True):
    class sortmixer(mixer):
        def __init__(self,deep=True):
            super(sortmixer, self).__init__()
            self.onstart = True
            if deep:
                from model.deep_sort_pytorch.deep_sort import DeepSort
                self.model = DeepSort(root_dir+"weights/deepsort/ckpt.t7")
            else:
                from model.sort import Sort
                self.model= Sort()
            self.update = self.model.update
            
        def preprocess(self,xyxyc):
            '''
            xyxyc->(xywh,c)
            '''
            xywhc = self.xyxyc2xywhc(xyxyc)
            return xywhc[:,:]
        
        def xyxyc2xywhc(self,j):
            j[:,2] = j[:,2] - j[:,0]
            j[:,3] = j[:,3] - j[:,1]
            return j[(j[:,2] != 0)&(j[:,3] != 0)]
        
        def inference(self,xywhc,im0):
            xywh = xywhc[:,:4]
            c = xywhc[:,4]
            if self.onstart:
                self.update(xywh,c,im0)
                self.update(xywh,c,im0)
                self.onstart = False
            return self.update(xywh,c,im0)

        
    return sortmixer(deep)
            


def get_deepmar():
    from model.deepmar.baseline.model.DeepMAR import DeepMAR_ResNet50
    import torchvision.transforms as transforms
    class deepmarmixer(mixer):
        def __init__(self):
            super(deepmarmixer, self).__init__()
            self.model = DeepMAR_ResNet50()
            self.model.load_state_dict(torch.load(
                root_dir+'weights/deepmar/deepmar_statedict.pth'))
            self.model.eval()
            self.attrs = pickle.load(
                open(root_dir+'model/deepmar/attribute_list_Chinese.pkl', "rb"))
            resize = 224
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            self.input_transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.Normalize(mean=mean, std=std),
            ])

        def preprocess(self,img,xyxyi):
            from PIL import Image
            '''
            img shape :[h,w,c](y,x,c)
            output: stack of sub-images,ids
            '''
            img = img[:,:,::-1].copy()
            img = torch.tensor(img,requires_grad=False).permute(2,0,1).float()
            xyxy = xyxyi[:,:4]
            # xyxy[:,2] = xyxy[:,0] + xyxy[:,2]
            # xyxy[:,3] = xyxy[:,1] + xyxy[:,3]
            imgs = []
            for a,b,c,d in xyxy:
                cur = self.input_transform(img[:,b:d,a:c])
                imgs.append(cur)
            return imgs,torch.tensor(xyxyi[:,4])
        
        @torch.no_grad()
        def inference(self,imgs)->torch.tensor:
            ans = torch.stack([self.model(img.unsqueeze(0)).squeeze(0) for img in imgs])
            return ans
             
        
        def postprocess(self,o:torch.tensor,ids,topk=5):
            o = torch.cat((ids.view(-1,1),o.topk(5)[1]),-1)
            return o


    return deepmarmixer()


def get_mnnet():
    from model.HydraPlusNet.api import transform, model, attrs

    class mixer():
        def __init__(self):
            self.model = model(
                root_dir+"weights/mnnet/MNet_epoch_995")  # 输入rgb
            self.model.eval()
            self.transform = transform
            self.attrs = attrs

    return mixer()


def get_facelib():
    from model.FaceLib.facelib import AgeGenderEstimator, FaceDetector, EmotionDetector

    class mixer():
        def __init__(self):
            self.align = FaceDetector(
                weight_path=root_dir+'weights/facelib/mobilenet0.25_Final.pth')
            self.agegender = AgeGenderEstimator(
                weight_path=root_dir+'weights/facelib/ShufflenetFull.pth')
            self.emotion = EmotionDetector(
                weight_path=root_dir+"weights/facelib/densnet121.pth")

    return mixer()


def get_faceattr():
    class opt:
        model_type = 'resnet50'
    from model.face_attribute_classification_pytorch.model import AttClsModel
    from model.face_attribute_classification_pytorch.utils.utils import torch_init_model
    from model.face_attribute_classification_pytorch.test import load_atts

    class mixer():
        def __init__(self):
            self.model = AttClsModel(opt)
            torch_init_model(self.model, root_dir+"weights/FAC/best_model.pth")
            self.model.eval()
            self.attrs = load_atts(
                root_dir + "model/face_attribute_classification_pytorch/data_list/att_map.txt")
    return mixer()


def get_pose_estimator():
    '''
    poser = get_pose_estimator()
    pose = poser.inference(poser.model,'fdgfdgfdg.png',[{'bbox':[0,0,220,445],'track_id':1}])
    '''
    from model.Lite_HRNet.api import get_model, convert_coco_to_openpose_cords
    import cv2
    config = root_dir + \
        "model/Lite_HRNet/configs/top_down/lite_hrnet/coco/litehrnet_18_coco_256x192.py"
    weights = root_dir+'weights/LiteHRNet/litehrnet_18_coco_256x192.pth'

    class mixer():
        def __init__(self):
            self.model, self.inferencer = get_model(config, weights)
            self.model.eval() 

        def inference(self, *args):
            '''
            model, img_or_path, person_results
            if img_size is not [192,256],call scaler before me
            '''
            out = self.inferencer(*args)[0]
            for i in out:
                i['keypoints'] = convert_coco_to_openpose_cords(i['keypoints'])
            return out

        def scaler(self, img):
            img = cv2.resize(img, (192, 256))
            return img
            # person_result like [{'bbox':[0,0,192,256],'track_id':1},...]
    return mixer()


def get_action_recognitioner():
    from model.mmskeleton.api import get_model, get_labels, pose_normal, inference

    class mixer():
        def __init__(self):
            self.model = get_model()
            self.model.eval()
            self.labels = get_labels()
            self.pose_normal = pose_normal
            self.inference = inference
    return mixer()

# print(pose)
