# coding=gbk
import cv2
import numpy as np
import shape_based_matching_py
from IPython import embed
import os
import json

prefix = "/home/shimr/shapa_match/shape_based_matching-python_binding/test/"



def rotateTemplate(img, rot_deg, scale):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -rot_deg, scale)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h*sin) + (w*cos))
        new_h = int((h*cos) + (w*sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(img, M, (new_w, new_h)), M

def read_json(path):
    with open(path,'r') as load_f:
        load_dict = json.load(load_f)
        points = load_dict['shapes'][0]['points'] 
    return load_dict,points 

def write_json(path,dict):
    with open(path,"w") as f:
        json.dump(dict,f)    

import shutil  
       

if __name__ == "__main__":
    #test_path = 'D:/download/pianyi/pianyi/pianyi/4/tem_101_20210305130120_11_4_298529_69249.bmp'
    #test_img = cv2.imread(test_path)
    total_time = []
    img_dir = '/home/shimr/shapa_match/shape_based_matching-python_binding/images/'
    label_dir = '/home/shimr/shapa_match/shape_based_matching-python_binding/images/'
    rot_img_dir = '/home/shimr/shapa_match/shape_based_matching-python_binding/image_rot/'
    rot_label_dir = '/home/shimr/shapa_match/shape_based_matching-python_binding/image_rot/'
    
    shutil.rmtree(rot_img_dir)  
    os.mkdir(rot_img_dir)   
    name_list = os.listdir(img_dir)
    print(name_list)
    for name in name_list:
        if "tem" in name or name.endswith('json'):
            continue
        fpath = os.path.join(img_dir,name)
        img = cv2.imread(fpath)
        label_path = os.path.join(label_dir,name.replace('bmp','json'))
        json_dict, points = read_json(label_path)
        points_o = np.array(points)
        points = np.concatenate( (points_o,np.array([1,1,1,1]).reshape(-1,1)),axis =1)  #transform to 4*3
        points = points.T # 3*4
        
        
        
        
        
        #embed()
        for angle in range(0,360,5):
            for scale in range(1,2,1):
                print(angle,scale)
                rot_img,M  = rotateTemplate(img,angle,scale) # M is 2*3
                print("rot_img",rot_img.shape)
                #_,M  = rotateTemplate(pad_img,angle,scale)
                rot_img_name = name.replace('.bmp','') + '_'+str(angle) + '_' + str(scale)+'.bmp'
                rot_img_path = rot_img_dir + rot_img_name
                
                trans_points = np.dot(M,points).T
                print("points",points,"trans_points",trans_points)
                #cv2.polylines(rot_img, np.int32([points_o]), True, (255, 255, 0),thickness=3)
                #cv2.polylines(rot_img, np.int32([trans_points]), True, (0, 255, 0),thickness=3)
                cv2.imwrite(rot_img_path,rot_img)
                
                
                json_dict_trans = json_dict.copy()
                json_dict_trans['shapes'][0]['points'] = trans_points.tolist()
                write_json(os.path.join(rot_label_dir,rot_img_name.replace('bmp','json')),json_dict_trans)
                
                
                
                
                
    print("label points",points)
