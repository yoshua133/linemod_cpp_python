# coding=gbk
import cv2
import numpy as np
import shape_based_matching_py
from IPython import embed
import os
import json
import time
import math

prefix = "/home/xiangdawei/linemod_python/linemod_cpp_python/result_save/"
prefix_visual = "/home/xiangdawei/linemod_python/linemod_cpp_python/result_visual/"
ori_img_path = "/home/xiangdawei/linemod_python/linemod_cpp_python/images/101_20210305130120_11_4_298529_69249.bmp"
temp_path = "/home/xiangdawei/linemod_python/linemod_cpp_python/tem_101_20210305130120_11_4_298529_69249.bmp"
img_dir = '/home/xiangdawei/linemod_python/linemod_cpp_python/image_rot/'


#from shapely import geometry
"""
def if_inPoly(polygon, Points):
    line = geometry.LineString(polygon)
    point = geometry.Point(Points)
    polygon = geometry.Polygon(line)
    return polygon.contains(point)
"""

def rotateTemplate(img, rot_deg, scale):
        img = img.copy()
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rot_deg, scale)  #-rot_deg, scale)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = w # int((h*sin) + (w*cos))  对于padding image的旋转是和原图像同样大小的
        new_h = h # int((h*cos) + (w*sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(img, M, (new_w, new_h)), M

def read_json(path):
    with open(path,'r') as load_f:
        load_dict = json.load(load_f)
        points = load_dict['shapes'][0]['points'] 
    return load_dict,points 
        
def check_right(pred,gt):
    pred = np.array(pred) #4*2
    gt = np.array(gt)#4*2
    #print("pred",pred,pred.shape)
    #print("gt",gt,gt.shape)
    num = 0
    for i in range(4):
        point = gt[i,:]
        #print("point",point,if_inPoly(pred,point))
        if if_inPoly(pred,point):
            num +=1
    return num 

def get_degree_center(box):
    #box must be 4*2
    center = np.array(box).mean(0)
    x1 = box[0,0]
    y1 = box[0,1]
    x2 = box[1,0]
    y2 = box[1,1]
    tan = (y2 - y1)/(x2-x1+1e-10)
    theta = math.atan(tan) *180 /3.14
    if theta <0:
        theta += 180
    return center,theta, tan
    
    
import shutil  
shutil.rmtree(prefix)  
os.mkdir(prefix)         
shutil.rmtree(prefix_visual)  
os.mkdir(prefix_visual)       
def train_test(mode, use_rot):
    detector = shape_based_matching_py.Detector(128, [1, 4]) #128是num_features [4,8]是T
        
    #img = cv2.imread(temp_path)
    ori_img = cv2.imread(ori_img_path)
    img = ori_img[88:176,76:191,] #标注范围
    
    # print(img.shape)

    # order of ny is row col
    #img = img[110:380, 130:400]
    mask = np.ones((img.shape[0], img.shape[1]), np.uint8)
    mask *= 255

    padding = 100
    padded_img = np.zeros((img.shape[0]+2*padding, 
        img.shape[1]+2*padding, img.shape[2]), np.uint8)
    padded_mask = np.zeros((padded_img.shape[0], padded_img.shape[1]), np.uint8)

    padded_img[padding:padded_img.shape[0]-padding, padding:padded_img.shape[1]-padding, :] = \
        img[:, :, :]
    padded_mask[padding:padded_img.shape[0]-padding, padding:padded_img.shape[1]-padding] = \
        mask[:, :]
    cv2.imwrite(prefix+"padded_temp_img.jpg", padded_img)
    cv2.imwrite(prefix+"padded_temp_mask.jpg", padded_mask)
    # cv2.waitKey()

    shapes = shape_based_matching_py.shapeInfo_producer(padded_img, padded_mask)
    shapes.angle_range = [0, 360]
    shapes.angle_step = 5
    shapes.scale_range = [1]
    shapes.produce_infos()  #shapes中有一个info 构成的vector，produce函数 通过range和step 重构infos
    #embed()
    infos_have_templ = []
    class_id = "test"
    is_first = True
    first_id = 0
    first_angle = 0
    angle_scale_map = dict()
    
    matrix_map = dict()
    print("total shape",len(shapes.infos))
    max_id = 0
    for info in shapes.infos:
        to_show = shapes.src_of(info) #info only have scale and angle
        # for each input image src, do a angle and scale transform  
        # this transform doesn't extend the border, so it is the same shape as the padding image
        
        templ_id = 0
        if is_first:
            templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info))  #mask of to rotate the input image
            # 如果return false的话 说明template image上没有多于4个的feature points，detector设定num_features=128
            #print("return temp id",templ_id)
            first_id = templ_id
            first_angle = info.angle

            if use_rot:
                is_first = False
        else:
            templ_id = detector.addTemplate_rotate(class_id, first_id,
                                                   info.angle-first_angle,
                shape_based_matching_py.CV_Point2f(padded_img.shape[1]/2.0, padded_img.shape[0]/2.0))  # 加入旋转template的时候是不extend的那种，旋转中心一直是padding_img的center
            # cpp源码这里是找到template_pyramids中的first id的template pyramid 然后对每个level下的feature point 以padding image的中心旋转，因为保存feature pyramids的其实是features
        print("angles",templ_id,info.angle)
        angle_scale_map[templ_id] = [info.angle, info.scale]
        _, M = rotateTemplate(padded_img,info.angle, info.scale)
        h, w = img.shape[:2]   #padded_img.shape[:2]
        corners_3xn = np.array([[0+padding, 0+padding, 1],
                                        [w+padding, 0+padding, 1],
                                        [w+padding, h+padding, 1],
                                        [0+padding, h+padding, 1]]).T
        #corners_3xn = np.array([[0, 0, 1],
        #                                [w, 0, 1],
        #                                [w, h, 1],
        #                                [0, h, 1]]).T
        new_corners_nx3 = np.dot(M, corners_3xn)
        new_corners_nx3 = new_corners_nx3 #+padding
        #print(templ_id,angle_scale_map[templ_id],M)
        #print("new_corners_nx3",np.int32(new_corners_nx3))
        #print("to_show",to_show.shape)
        matrix_map[templ_id] = new_corners_nx3
        
        
        #print(class_id,templ_id)
        templ = detector.getTemplates(class_id, templ_id)
        #print("templ[0].tl_x",templ[0].tl_x,templ[0].tl_y)
        for feat in templ[0].features:
            to_show = cv2.circle(to_show, (feat.x+templ[0].tl_x, feat.y+templ[0].tl_y), 3, (0, 0, 255), -1)
            
        pts = list()
        for j in range(4):
                pts.append([new_corners_nx3[0,j], new_corners_nx3[1,j] ])
        cv2.polylines(to_show, np.int32([pts]), True, (255, 255, 255),thickness=3)
        
        cv2.imwrite(prefix+"temp_{}.jpg".format(info.angle), to_show)
        #cv2.waitKey(1)
        if templ_id != -1:
            infos_have_templ.append(info)
        if templ_id > max_id:
            max_id = templ_id
    detector.writeClasses(prefix+"case1/%s_templ.yaml")
    shapes.save_infos(infos_have_templ, prefix + "case1/test_info.yaml")
    print("max_id",max_id)

    # test
    
    ids = []
    ids.append('test')

    producer = shape_based_matching_py.shapeInfo_producer()
    infos = producer.load_infos(prefix + "case1/test_info.yaml")
    
    
    errors = []
    name_list = os.listdir(img_dir)
    test_img_num = 0
    num_of_detected =  0
    total_time = []
    simi = []
    error_name_dict = dict()
    for name in name_list:
        if "tem" in name or name.endswith('json') or not "305" in name.split('_')[-2]:
            continue
        print(
        name)
        test_img_angle = name.split('_')[-2]
        fpath = os.path.join(img_dir,name)
        test_img_num +=1
        if test_img_num>10000:
            continue
        json_path = fpath.replace('bmp','json')
        _, rect_label_o = read_json(json_path)
        
        test_img = cv2.imread(fpath)#prefix+"case1/test.png")
        padding = 250
        padded_img = np.zeros((test_img.shape[0]+2*padding, 
            test_img.shape[1]+2*padding, test_img.shape[2]), np.uint8)
        padded_img[padding:padded_img.shape[0]-padding, padding:padded_img.shape[1]-padding, :] = \
            test_img[:, :, :]
    
        stride = 16
        img_rows = int(padded_img.shape[0] / stride) * stride
        img_cols = int(padded_img.shape[1] / stride) * stride
        img = np.zeros((img_rows, img_cols, padded_img.shape[2]), np.uint8)
        img[:, :, :] = padded_img[0:img_rows, 0:img_cols, :]
        start = time.time()
        print("in name",name.strip('.bmp'))
        matches = detector.match(img, 20, name.strip('.bmp'), ids)
        exc_time = time.time() - start
        total_time.append(exc_time)
        #embed()
        top5 = 1
        if top5 > len(matches):
            top5 = 1

        rect_label = np.array(rect_label_o) + padding
        for i in range(1):
            if len(matches) < 1:
                print("no match")
                continue
            num_of_detected +=1
            match = matches[i]
            templ = detector.getTemplates("test", match.template_id)
            
            #to get the four points of the rect in the template add match_x draw a poly 
            matrix_i = matrix_map[match.template_id]
            angle_i = angle_scale_map[match.template_id][0]
            #print("matrix_i",matrix_i)
            pts = list()
            for j in range(4):
                    pts.append([matrix_i[0,j] + match.x -templ[0].tl_x, matrix_i[1,j] + match.y -templ[0].tl_y])   # template 的tl.x 应该保存的是feature的xy相对padding之后模板的一个关系
            cv2.polylines(img, np.int32([pts]), True, (128, 255, 128))
            cv2.polylines(img, np.int32([rect_label.tolist()]), True, (0, 255, 0))
            #print("""
            #""")
            #print("pred inside gt",check_right(pts,rect_label))
            
            center_pred,theta_pred,tan0 = get_degree_center(np.array(pts))
            center_gt,theta_gt,tan1 = get_degree_center(np.array(rect_label))
            offset_center = np.sum(np.abs(center_pred-center_gt))
            offset_theta = np.abs(theta_pred-  theta_gt)
            #print("pts",pts,"rect_label",rect_label,"center_pred",center_pred,"theta_pred",theta_pred,"center_gt",center_gt,"theta_gt",theta_gt,"offset_center",offset_center,"offset_theta",offset_theta,"angle_i",angle_i,"tan0",tan0,"tan1",tan1)
            errors.append([offset_center,offset_theta,float(test_img_angle)-(360-float(angle_i))])
            error_name_dict[test_img_angle] = [offset_center,offset_theta,float(test_img_angle)-(360-float(angle_i))]
            #embed()
            # r_scaled = 270/2.0*infos[match.template_id].scale
            # train_img_half_width = 270/2.0 + 100
            # train_img_half_height = 270/2.0 + 100
            # x =  match.x - templ[0].tl_x + train_img_half_width
            # y =  match.y - templ[0].tl_y + train_img_half_height
            for feat in templ[0].features:
                img = cv2.circle(img, (feat.x+match.x, feat.y+match.y), 3, (0, 0, 255), -1)
            
            # cv2 have no RotatedRect constructor?
            print('match.template_id: {}'.format(match.template_id))
            print('match.similarity: {}'.format(match.similarity))
            simi.append(match.similarity)
            #print("rect_label",rect_label)
            #print("rect_label_o",rect_label_o)
            #print("pred pts",pts)
            cv2.imwrite(prefix+"img_test"+name, img)
            #cv2.waitKey(0)
    errors = np.abs(np.array(errors))
    print(np.int32(errors),"""
    """,error_name_dict,
    np.sum(errors[:,0]>4),np.sum(errors[:,1]>4),test_img_num, "mean error offset",np.mean(errors[:,0]), "mean error theta offset",np.mean(errors[:,1]) )
    print("simi",simi)
    print("simi <60",np.where(np.array(simi)<85))
    print("average time",sum(total_time)/float(len(total_time)))
    print("num_of_detected",num_of_detected)

if __name__ == "__main__":
    train_test('train', True)
    
