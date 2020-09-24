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
import torch
import torch.nn.utils.prune as prune
import time

PRUNE=True

def main():
    config = '/media/shalev/98a3e66d-f664-402a-9639-15ec6b8a7150/work_dirs/try2/faster_rcnn_r50_caffe_c4_1x_coco_shalev.py'
    checkpoint = '/media/shalev/98a3e66d-f664-402a-9639-15ec6b8a7150/work_dirs/try2/latest.pth'
    src_img_path = '/home/shalev/downloads/1pic_coco/000000000285.jpg'
    dst_img_path = '/home/shalev/downloads/1pic_coco/000000000285_res.jpg'
    img = cv2.imread(src_img_path)
    model = mmdet.apis.init_detector(config, checkpoint=checkpoint, device='cuda:0')
    for i in range(10):
        if PRUNE:
            # backbone = model.backbone
            modules = [model.backbone.children(), model.roi_head.children(), model.rpn_head.children()]
            for main_module in modules:
                for module in main_module:
                    if isinstance(module,torch.nn.Conv2d) or isinstance(module,torch.nn.Linear):
                        print("before: ",module.weight.sum())
                        prune.ln_structured(module, name='weight', amount=0.05, dim=0, n=float('-inf'))
                        print("after: ", module.weight.sum())
                    else:
                        for sub in module.children():
                            if isinstance(sub, torch.nn.Conv2d) or isinstance(sub,torch.nn.Linear):
                                print("before: ", sub.weight.sum())
                                prune.ln_structured(sub, name='weight', amount=0.15, dim=0, n=float('-inf'))
                                print("after: ", sub.weight.sum())
                            else:
                                for sub_sub in sub.children():
                                    if isinstance(sub_sub, torch.nn.Conv2d) or isinstance(sub_sub,torch.nn.Linear):
                                        print("before: ", sub_sub.weight.sum())
                                        prune.ln_structured(sub_sub, name='weight', amount=0.15, dim=0, n=float('-inf'))
                                        print("after: ", sub_sub.weight.sum())

        start = time.time()
        res=mmdet.apis.inference_detector(model, img)
        print("Inference time: ",(time.time()-start))
        if hasattr(model, 'module'):
            model = model.module
        img_res = model.show_result(img, res, score_thr=0.305, show=True)
    # cv2.imwrite(dst_img_path,img_res)


main()