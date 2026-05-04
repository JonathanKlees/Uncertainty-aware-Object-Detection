from pycocotools.coco import COCO
from my_cocoeval import COCOeval

import numpy as np
import argparse
import json

"""global parameters"""  
dataset = "COCO"  

boxthresh = 0.01
textthresh = 0.2
nms = 0.6

eval_iou_thresh = 0.5

gt_path = f"/path/to/benchmark_datasets_gt/{dataset}/val_coco_format.json"
dt_path = f"/path/to/benchmark_datasets_predictions_second_round/GroundingDino/{dataset}/boxthresh_{boxthresh}_textthresh_{textthresh}/nms_{nms}/val_coco_format.json"

def eval_vals():
    """load ground truth"""
    coco_gt = COCO(gt_path) 
    
    if "info" not in coco_gt.dataset:
        coco_gt.dataset["info"] = {}

    if "licenses" not in coco_gt.dataset:
        coco_gt.dataset["licenses"] = []
    
    """load predictions"""
    #coco_dt = coco_gt.loadRes(dt_path)
    
    coco_dt = coco_gt.loadRes([p for p in json.load(open(dt_path)) if p["image_id"] in set(coco_gt.getImgIds())])
    
    """evaluation"""
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    
    #coco_eval.params.imgIds = ["15"]
    coco_eval.params.iouThrs = np.linspace(eval_iou_thresh, eval_iou_thresh, int(np.round((eval_iou_thresh - eval_iou_thresh) / .05)) + 1, endpoint=True)
    coco_eval.params.areaRng = [[0 ** 2, 1e5 ** 2]]
    coco_eval.params.maxDets = [100]
    coco_eval.params.areaRngLbl = ['all']

    coco_eval.evaluate()
    tps, fps, fns, precision, recall, f1 = coco_eval.accumulate_tp_fp_fn()
    
    gt_objs = tps + fns
    pred_objs = tps + fps
    
    """output"""
    print(f"{dataset} - IoU thresh: {eval_iou_thresh}")
    print(f"\nNumber of GT objects: {gt_objs}")
    print(f"Number of predictions: {pred_objs}")
    print(f"\nTPs: {tps}")
    print(f"FPs: {fps}")
    print(f"FNs: {fns}")
    print(f"\nPrecision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}\n")
    

def eval_ap():
    """load ground truth"""
    coco_gt = COCO(gt_path) 
    
    if "info" not in coco_gt.dataset:
        coco_gt.dataset["info"] = {}

    if "licenses" not in coco_gt.dataset:
        coco_gt.dataset["licenses"] = []
        
    
    """load predictions"""
    #coco_dt = coco_gt.loadRes(dt_path)
    coco_dt = coco_gt.loadRes([p for p in json.load(open(dt_path)) if p["image_id"] in set(coco_gt.getImgIds())])

    """evaluation"""
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", help="Evaluation of predictions", action="store_true")
    parser.add_argument("--eval_vals", help="Calculate TPs, FPs and FNs", action="store_true")
    
    args = parser.parse_args()
    
    if args.eval:
        eval_ap()
    elif args.eval_vals:
        eval_vals()
