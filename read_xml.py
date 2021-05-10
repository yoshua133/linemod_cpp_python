import cv2
import numpy as np
from IPython import embed

def read(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    mapNode = fs.getNode('vocabulary')
    mat = mapNode.mat()
    return mat

path1 = "/home/xiangdawei/linemod_python/linemod_cpp_python/result_visual/101_20210305130120_11_4_298529_69249_305_1quantized.xml"
path2 = "/home/xiangdawei/linemod_python/linemod_cpp_python/result_visual/fist_simi.xml"
path3 = "/home/xiangdawei/linemod_python/linemod_cpp_python/result_visual/101_20210305130120_11_4_298529_69249_305_1spread.xml"


embed()
