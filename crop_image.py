import os 
import cv2
from xml.dom import minidom
import numpy as np


def made(idx,path):
    true_box = []
    dom=minidom.parse(path)
    names=dom.getElementsByTagName('name')
    xmin =dom.getElementsByTagName('xmin')
    ymin =dom.getElementsByTagName('ymin')
    xmax =dom.getElementsByTagName('xmax')
    ymax =dom.getElementsByTagName('ymax')

    true_box.append(int(names[idx].firstChild.data))
    true_box.append(int(xmin[idx].firstChild.data))
    true_box.append(int(ymin[idx].firstChild.data))
    true_box.append(int(xmax[idx].firstChild.data))
    true_box.append(int(ymax[idx].firstChild.data))
    return true_box

xml_path = '/data/haoyuan/YOLOX/datasets/VOCdevkit/VOC2007/Annotations/'
img_path = '/data/haoyuan/YOLOX/datasets/VOCdevkit/VOC2007/JPEGImages/'
file_load = os.listdir(xml_path)

for i in range(143):
    if i ==-1:
        pass
    else:
        num = 0
        os.mkdir('/data/haoyuan/MobileNetV3-master/MobileNetV3-master/data/splitData/train/'+str(i))
        os.mkdir('/data/haoyuan/MobileNetV3-master/MobileNetV3-master/data/splitData/valid/'+str(i))
        for file in file_load:
            # base = os.path.basename(file)
            jpg_name = file.split('.xml')[0]+'.jpg'
            dom=minidom.parse(xml_path+file)
            obj = dom.getElementsByTagName("object")
            for k in range(len(obj)):
                t = made(k,xml_path+file)
                if t[0]==i:
                    im = cv2.imread(img_path+jpg_name)
                    height = im.shape[0]
                    width = im.shape[1]
                    if t[2]-10 < 0:
                        a = t[2]
                    else:
                        a = t[2]-10
                    if t[4]+10 > height:
                        b = t[4]
                    else:
                        b = t[4]+10
                    if t[1]-10 < 0:
                        c = t[1]
                    else:
                        c = t[1]-10
                    if t[3]+10 > width:
                        d = t[3]
                    else:
                        d = t[3]+10
                    im_crop = im[a:b,c:d]
                    num = num + 1
                    if num % 5 == 0:
                        cv2.imwrite('/data/haoyuan/MobileNetV3-master/MobileNetV3-master/data/splitData/valid/'+str(i)+'/'+jpg_name,im_crop)
                    else:
                        cv2.imwrite('/data/haoyuan/MobileNetV3-master/MobileNetV3-master/data/splitData/train/'+str(i)+'/'+jpg_name,im_crop)
print("successful")
                
            

