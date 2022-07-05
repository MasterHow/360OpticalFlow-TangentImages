import os, sys
# to import module in sibling folders
dir_scripts = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_scripts + "/src") #  code/python/src 

DATA_DIR = dir_scripts + "/data/"

import image_io
import flow_estimate
import flow_io
import flow_vis
import flow_postproc
import flow_evaluate
import torch
import cv2

import numpy as np
import dataset as dataset
import time
from logger import Logger
log = Logger(__name__)
log.logger.propagate = False


def validate_replica(data_root, timing=False, valid=False):
    """Peform validation using the Replica360 for this method"""
    results = {}
    epe_any = []
    if timing:
        time_list_all = []

    for dstype in ['line', 'rand', 'circ']:
        val_dataset = dataset.Replica360(root=data_root, dstype=dstype)
        epe_list = []

        if timing:
            time_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, valid_mask = val_dataset[val_id]
            image1 = cv2.cvtColor(image1.squeeze(dim=0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8),
                                  cv2.COLOR_RGB2BGR)
            image2 = cv2.cvtColor(image2.squeeze(dim=0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8),
                                  cv2.COLOR_RGB2BGR)

            if timing:
                torch.cuda.synchronize()
                time_start = time.time()
            flow, flow_vis = infer_with_tangent(image1, image2)
            if timing:
                torch.cuda.synchronize()
                time_end = time.time()
                time_sum = time_end - time_start
                print('*'*10, 'infer_time:', time_sum, 's', '*'*10)
                time_list.append(time_sum)

            flow_gt = flow_gt.numpy().transpose(1, 2, 0)

            if valid:
                large_valid = (flow_gt[:, :, 0] > 1e7) | (flow_gt[:, :, 1] > 1e7)
                non_valid = (flow_gt[:, :, 0] == np.nan) | (flow_gt[:, :, 1] == np.nan)
                valid_mask = (~non_valid & ~large_valid).astype(np.uint8) * valid_mask.numpy()

                epe = np.sum(np.sqrt(np.sum((flow - flow_gt) ** 2, axis=2)) * valid_mask) / np.sum(valid_mask)
            else:
                epe = np.sum(np.sqrt(np.sum((flow - flow_gt) ** 2, axis=2)))/(flow.shape[0]*flow.shape[1])

                # original epe func
                # 3) error metric
                # epe = flow_evaluate.EPE(flow_gt, flow)

            print('*' * 10, 'epe:', epe, 'px', '*' * 10)
            epe_list.append(epe)

        epe_mean = sum(epe_list) / len(epe_list)

        if timing:
            infer_time = np.mean(time_list)
            time_list_all += time_list
            print('Validation Replica (%s) EPE: %f, Time: %f' %
                  (dstype, epe_mean, infer_time))
        else:
            print('Validation Replica (%s) EPE: %f' %
                  (dstype, epe_mean))

        epe_any.append(epe_list)

    epe_final_all = np.concatenate(epe_any)
    epe_final = np.mean(epe_final_all)
    if timing:
        infer_time_all = np.mean(time_list_all)
        print('Validation Replica (all) EPE: %f, Time: %f' %
              (epe_final, infer_time_all))
    else:
        print('Validation Replica (all) EPE: %f' %
              (epe_final))

    return results


def infer_with_tangent(image1_erp, image2_erp, padding_size=0.3):
    src_erp_image = image1_erp
    tar_erp_image = image2_erp

    # 1) estimate optical flow
    flow_estimator = flow_estimate.PanoOpticalFlow()
    flow_estimator.debug_enable = False
    flow_estimator.debug_output_dir = None
    flow_estimator.padding_size_cubemap = padding_size
    flow_estimator.padding_size_ico = padding_size
    flow_estimator.flow2rotmat_method = "3D"
    flow_estimator.tangent_image_width_ico = 480
    optical_flow = flow_estimator.estimate(src_erp_image, tar_erp_image)

    # 2) evaluate the optical flow and output result
    # output optical flow image
    optical_flow = flow_postproc.erp_of_wraparound(optical_flow)
    optical_flow_vis = flow_vis.flow_to_color(optical_flow, min_ratio=0.2, max_ratio=0.8)

    return optical_flow, optical_flow_vis


def test_tangent(data_root=None, timing=False, valid=False):
    validate_replica(data_root, timing, valid)
    return 0


if __name__ == "__main__":

    test_tangent(data_root='H://Replica360', timing=True, valid=False)
