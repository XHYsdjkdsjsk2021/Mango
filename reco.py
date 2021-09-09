import cv2 
import numpy as np
from image_similarity_function import *
import os
import shutil
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms

class FeatureExtractor(nn.Module): # 提取特征工具
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
 
    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": 
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs

model = models.resnet50(pretrained=True) # 加载resnet50工具
model = model.cuda()
model.eval()

def input_image(img_standard):
    img_standard=cv2.resize(img_standard,(224,224));
    img_standard=cv2.cvtColor(img_standard,cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_standard=transform(img_standard).cuda()
    img_standard=img_standard.unsqueeze(0)
    model2 = FeatureExtractor(model, ['layer4']) # 指定提取 layer3 层特征
    with torch.no_grad():
        out_=model2(img_standard)
        # print(len(out_), out_[0].shape)
    return out_

def cosine_similarity(x, y, dim=256):
    xx = 0.0
    yy = 0.0
    xy = 0.0
    for i in range(dim):
        xx += x[i] * x[i]
        yy += y[i] * y[i]
        xy += x[i] * y[i] 
    xx_sqrt = xx ** 0.5
    yy_sqrt = yy ** 0.5
    cos = xy/(xx_sqrt*yy_sqrt)*0.5+0.5
    return cos

img1_path = '/data/haoyuan/audit_data/logo_test/2021_08_31_10_18_10/00103.jpg'
img2_path = '/data/haoyuan/audit_data/logo_test/2021_08_31_10_18_10/00350.jpg'
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
height = img1.shape[0] #1080
width = img1.shape[1]  #1920
def get_location(mask):
    top = np.min(np.where(mask)[0]) 
    left = np.min(np.where(mask)[1])
    down = np.max(np.where(mask)[0])
    right = np.max(np.where(mask)[1])
    return top ,left,down,right

def detect(img1,img2,height_cor,width_cor):
    gray_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    edge_1 = cv2.Canny(gray_1, 100, 300)
    edge_2 = cv2.Canny(gray_2, 100, 300)
    # cv2.imwrite('/data/haoyuan/1.jpg',edge_1)
    # cv2.imwrite('/data/haoyuan/2.jpg',edge_2)
    small_height = int(height*height_cor)
    small_width = int(width*width_cor)
    mask = np.zeros([small_height, small_width], dtype=np.uint8)
    return mask,small_height,small_width,edge_1,edge_2,img1
# 右上
img1_right = img1[0:int(height*0.2),int(width*0.8):width]
img2_right = img2[0:int(height*0.2),int(width*0.8):width]

#左上
img1_left = img1[0:int(height*0.2),0:int(width*0.2)]
img2_left = img2[0:int(height*0.2),0:int(width*0.2)]

#左下
img1_left_down = img1[int(height*0.6):height,0:int(width*0.2)]
img2_left_down = img2[int(height*0.6):height,0:int(width*0.2)]

#右下
img1_right_down = img1[int(height*0.6):height,int(width*0.8):width]
img2_right_down = img2[int(height*0.6):height,int(width*0.8):width]

num = 0
mask,small_height,small_width,edge_1,edge_2,img_or = detect(img1_right,img2_right,0.2,0.2)
for row in range(small_height):
    for col in range(small_width):
        if edge_1[row][col] == edge_2[row][col] and edge_1[row][col] != 0:
            mask[row][col] = 255
            num = num + 1
if num < 50:
    print('rightup=None')
else:
    top ,left,down,right = get_location(mask)
    img_or = img_or[top:down,left:right]
    # img1_path = '/data/haoyuan/leftup.jpg'
    # 搜索文件夹
    filepath='/data/haoyuan/Standard/'
    kk_ = 0
    for parent, dirnames, filenames in os.walk(filepath):
        for filename_ in filenames:
            # print(filepath+filename)
            img2_path=filepath+filename_
            out = input_image(img_or)
            img_standard = cv2.imread(img2_path)
            out_ = input_image(img_standard)
            cos = cosine_similarity(out[0][0],out_[0][0],dim=2048)
            b = sum(sum(cos))
            re = b /(cos.shape[0]*cos.shape[1])
            print(filename_,re)
            if re > kk_:
                kk_ = re
                final = filename_
    print('rightup=',final)
        

num = 0
mask_left,small_height_left,small_width_left,edge_1_left,edge_2_left,or_left = detect(img1_left,img2_left,0.2,0.2)
for row in range(small_height_left):
    for col in range(small_width_left):
        if edge_1_left[row][col] == edge_2_left[row][col] and edge_1_left[row][col] != 0:
            mask_left[row][col] = 255
            num = num + 1
if num <50:
    print('leftup=None')
else:
    top ,left,down,right = get_location(mask_left)
    or_left = or_left[top:down,left:right]
    filepath='/data/haoyuan/Standard/'
    kk_ = 0
    for parent, dirnames, filenames in os.walk(filepath):
        for filename_ in filenames:
            # print(filepath+filename)
            img2_path=filepath+filename_
            out = input_image(or_left)
            img_standard = cv2.imread(img2_path)
            out_ = input_image(img_standard)
            cos = cosine_similarity(out[0][0],out_[0][0],dim=2048)
            b = sum(sum(cos))
            re = b /(cos.shape[0]*cos.shape[1])
            if re > kk_:
                kk_ = re
                final = filename_
            print(filename_,re)
    print('leftup=',final)



num = 0
mask_left_down,small_height_left_down,small_width_left_down,edge_1_left_down,edge_2_left_down,or_left_down = detect(img1_left_down,img2_left_down,0.4,0.2)
for row in range(small_height_left_down):
    for col in range(small_width_left_down):
        if edge_1_left_down[row][col] == edge_2_left_down[row][col] and edge_1_left_down[row][col] != 0:
            mask_left_down[row][col] = 255
            num = num + 1
if num < 50:
    print('left_down=None')
else:
    top ,left,down,right = get_location(mask_left_down)
    or_left_down = or_left_down[top:down,left:right]
    filepath='/data/haoyuan/Standard/'
    kk_ = 0
    for parent, dirnames, filenames in os.walk(filepath):
        for filename_ in filenames:
            # print(filepath+filename)
            img2_path=filepath+filename_
            out = input_image(or_left_down)
            img_standard = cv2.imread(img2_path)
            out_ = input_image(img_standard)
            cos = cosine_similarity(out[0][0],out_[0][0],dim=2048)
            b = sum(sum(cos))
            re = b /(cos.shape[0]*cos.shape[1])
            if re > kk_:
                kk_ = re
                final = filename_
    print('leftdown=',final)


num = 0
mask_right_down,small_height_right_down,small_width_right_down,edge_1_right_down,edge_2_right_down,or_right_down = detect(img1_right_down,img2_right_down,0.4,0.2)
for row in range(small_height_right_down):
    for col in range(small_width_right_down):
        if edge_1_right_down[row][col] == edge_2_right_down[row][col] and edge_1_right_down[row][col] != 0:
            mask_right_down[row][col] = 255
            num = num + 1
if num < 50:
    print('right_down=None')
else:
    top ,left,down,right = get_location(mask_right_down)
    or_right_down = or_right_down[top:down,left:right]
    filepath='/data/haoyuan/Standard/'
    kk_ = 0
    for parent, dirnames, filenames in os.walk(filepath):
        for filename_ in filenames:
            # print(filepath+filename)
            img2_path=filepath+filename_
            out = input_image(or_right_down)
            img_standard = cv2.imread(img2_path)
            out_ = input_image(img_standard)
            cos = cosine_similarity(out[0][0],out_[0][0],dim=2048)
            b = sum(sum(cos))
            re = b /(cos.shape[0]*cos.shape[1])
            if re > kk_:
                kk_ = re
                final = filename_
    print('rightdown=',final)




