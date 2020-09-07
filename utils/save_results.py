# import warnings
#
# import matplotlib.pyplot as plt
# import mmcv
# import numpy as np
# import torch
# from mmcv.ops import RoIAlign, RoIPool
# from mmcv.parallel import collate, scatter
# from mmcv.runner import load_checkpoint
#
# from mmdet.core import get_classes
# from mmdet.datasets.pipelines import Compose
# from mmdet.models import build_detector
import cv2
import mmdet.apis


def main():
    config = '/media/shalev/98a3e66d-f664-402a-9639-15ec6b8a7150/work_dirs/try1/faster_rcnn_r50_caffe_c4_1x_coco_shalev.py'
    checkpoint = '/media/shalev/98a3e66d-f664-402a-9639-15ec6b8a7150/work_dirs/try1/latest.pth'
    src_img_path = '/home/shalev/downloads/1pic_coco/000000000285.jpg'
    dst_img_path = '/home/shalev/downloads/1pic_coco/000000000285_res.jpg'
    img = cv2.imread(src_img_path)
    model=mmdet.apis.init_detector(config, checkpoint=checkpoint, device='cuda:0')
    res=mmdet.apis.inference_detector(model, img)
    if hasattr(model, 'module'):
        model = model.module
    img_res = model.show_result(img, res, score_thr=0.5, show=False)
    cv2.imwrite(dst_img_path,img_res)


main()