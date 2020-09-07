from mmdet.datasets import PIPELINES
import numpy as np

@PIPELINES.register_module()
class Duplicate:

    def __call__(self, results):
        results['img'] = np.dstack((results['img'],results['img']))
        results['img_shape'] = results['img'].shape
        results['ori_shape'][-1] *= 2

        return results