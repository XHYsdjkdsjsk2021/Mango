import os
import xml.dom.minidom
from PIL import Image
def imgRename(img_path):
    img_ori = os.listdir(img_path)
    count = 0
    for img in img_ori:
        old_path = os.path.join(img_path,img)
        new_name = ("%3d"%count).strip()+'.jpg'
        new_name = str('yolov5_')+new_name
        new_path = os.path.join(img_path,new_name)
        os.rename(old_path,new_path)
        count+=1
        print(old_path,new_path)



def genXml(img_path,xml_path):
    for img_file in os.listdir(img_path):
        img_name = os.path.splitext(img_file)[0]
        img = Image.open(os.path.join(img_path,img_file))
        img_size = img.size
        #create an empty dom document object
        doc = xml.dom.minidom.Document()
        #creat a root node which name is annotation
        annotation = doc.createElement('annotation')
        #add the root node to the dom document object
        doc.appendChild(annotation)
    
        #add the folder subnode
        folder = doc.createElement('folder')
        folder_text = doc.createTextNode('images')
        folder.appendChild(folder_text)
        annotation.appendChild(folder)
    
        #add the filename subnode
        filename = doc.createElement('filename')
        filename_text = doc.createTextNode(img_file)
        filename.appendChild(filename_text)
        annotation.appendChild(filename)
    
        # add the path subnode
        path = doc.createElement('path')
        path_text = doc.createTextNode('/data/haoyuan/YOLOX/datasets/VOCdevkit/VOC2007/JPEGImages' + img_file)
        path.appendChild(path_text)
        annotation.appendChild(path)
    
        #add the source subnode
        source = doc.createElement('source')
        database = doc.createElement('database')
        database_text = doc.createTextNode('Unknown')
        source.appendChild(database)
        database.appendChild(database_text)
        annotation.appendChild(source)
    
        #add the size subnode
        size = doc.createElement('size')
        width = doc.createElement('width')
        width_text = doc.createTextNode(str(img_size[0]))
        height = doc.createElement('height')
        height_text = doc.createTextNode(str(img_size[1]))
        depth = doc.createElement('depth')
        depth_text = doc.createTextNode('3')
        size.appendChild(width)
        width.appendChild(width_text)
        size.appendChild(height)
        height.appendChild(height_text)
        size.appendChild(depth)
        depth.appendChild(depth_text)
        annotation.appendChild(size)
    
        #add the segmented subnode
        segmented = doc.createElement('segmented')
        segmented_text = doc.createTextNode('0')
        segmented.appendChild(segmented_text)
        annotation.appendChild(segmented)
    
        #write into the xml text file
        # os.mknod(xml_path+'%s.xml'%img_name)
        print(xml_path+img_name)
        fp = open(os.path.join(xml_path,'%s.xml'%img_name), 'w+')
        doc.writexml(fp, indent='\t', addindent='\t', newl='\n')
        fp.close()
def genTxt(img_path,txt_path):
    img_names = os.listdir(img_path)
    for imgname in img_names:
        f = open(os.path.join(txt_path,imgname.split('.')[0]+'.txt'),"w")
        f.close()
    

img_path = '/data/haoyuan/yolov5_add'
xml_path = '/data/haoyuan/xml_label'
txt_path = '/data/haoyuan/txt_label'
imgRename(img_path)
# genXml(img_path,xml_path)
genTxt(img_path,txt_path)