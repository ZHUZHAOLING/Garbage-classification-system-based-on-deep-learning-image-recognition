from xml.dom.minidom import parse
import os
from glob import glob
import cv2 as cv
import json
from tqdm import tqdm


def readXML(xml_path):
    '''
    输入xml的地址， 返回对应图像的bbox_list [('168', '248', '851', '449'), ('198', '467', '533', '567')]
    '''
    domTree = parse(xml_path)
    # 文档根元素
    rootNode = domTree.documentElement
    # 所有目标
    object = rootNode.getElementsByTagName("object")
    img_name = rootNode.getElementsByTagName("filename")[0].childNodes[0].data
    # {'书籍纸张': [('447', '328', '3443', '2757')]} 类似于这种形式
    bbox_dict = {}
    for obj in object:
        bboxs = obj.getElementsByTagName("bndbox")
        name = obj.getElementsByTagName("name")[0].childNodes[0].data

        for bbox in bboxs:
            xmin = bbox.getElementsByTagName("xmin")[0].childNodes[0].data
            ymin = bbox.getElementsByTagName("ymin")[0].childNodes[0].data
            xmax = bbox.getElementsByTagName("xmax")[0].childNodes[0].data
            ymax = bbox.getElementsByTagName("ymax")[0].childNodes[0].data
            this_bbox = (xmin, ymin, xmax, ymax)
            this_bbox = [int(pos) for pos in this_bbox]
            if name not in bbox_dict:
                bbox_dict[name] = [this_bbox]
            else:
                bbox_dict[name].append(this_bbox)
    # print(bbox_dict)
    return img_name, bbox_dict

def zh_ch(string):
    return string.encode("gbk").decode('UTF-8', errors='ignore')

def resize_img_keep_ratio(img,target_size):
    old_size= img.shape[0:2]
    #ratio = min(float(target_size)/(old_size))
    ratio = min(float(target_size[i])/(old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i*ratio) for i in old_size])
    img = cv.resize(img,(new_size[1], new_size[0]), interpolation=cv.INTER_CUBIC)
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top,bottom = pad_h//2, pad_h-(pad_h//2)
    left,right = pad_w//2, pad_w -(pad_w//2)
    img_new = cv.copyMakeBorder(img,top,bottom,left,right, cv.BORDER_CONSTANT,None,(0,0,0))
    return img_new

if __name__ == '__main__':
    voc_path = r'F:\dosth\trash_trainval\trainval\VOC2007'
    annotations_path = os.path.join(voc_path, 'Annotations')
    img_path = os.path.join(voc_path, 'JPEGImages')
    new_trainval_path = r'F:\dosth\trash_trainval\trainval\new_trainval_from_voc'
    classes = r'F:\dosth\trash_trainval\trainval\train_classes_withoutFC.txt'
    class_dict = {}
    with open(classes,'r', encoding='utf-8' ) as f:
        class_info = f.readlines()
        for i, s_class in enumerate(class_info):
            # class_dict[s_class.strip()] = str(i)
            class_dict[str(i)] = s_class.strip()
    xmls = glob(os.path.join(annotations_path, '*.xml'))
    # nums_dict = { num:0 for num in range(44)} # 记录一下每个类别的图片数量
    num = 0
    with open('class.json', 'w') as f:
        json.dump(class_dict, f, indent=4, ensure_ascii=False)

    # for xml_file in tqdm(xmls):
    #     img_name, cls_dict = readXML(xml_file)
    #     ori_img = cv.imread(os.path.join(img_path, img_name))
    #     try:
    #         for this_class, pos_list in cls_dict.items():
    #             for pos in pos_list:
    #                 # print(pos)
    #                 x1, y1, x2, y2 = pos
    #                 new_img = ori_img[y1:y2, x1:x2]
    #                 h, w, c = new_img.shape
    #                 # new_img = new_img[::4, ::4]
    #                 # cv.waitKey(0)
    #                 if not (h < 200 and w < 200):
    #                     new_img = resize_img_keep_ratio(new_img, [600, 600])
    #                     # 写入图片和label
    #                     save_img_name = class_dict[this_class]+'_'+str(num)+'.jpg'
    #                     new_img_path = os.path.join(new_trainval_path, save_img_name)
    #                     cv.imwrite(new_img_path, new_img)
    #                     with open(new_img_path.replace('.jpg', '.txt'), 'w', encoding='utf-8') as f:
    #                         f.write(save_img_name.split(',')[0]+','+ class_dict[this_class])
    #                     num += 1
    #     except:
    #         pass






