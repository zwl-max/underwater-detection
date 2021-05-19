import time, argparse
from tqdm import tqdm
import os, cv2
import json
import mmcv
import glob 
import torch
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmcv.runner import wrap_fp16_model, load_checkpoint
from mmcv.cnn import fuse_conv_bn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from ensemble_boxes import *

def parse_args():
    parser = argparse.ArgumentParser(description='json2submit_nms')
    parser.add_argument('--jsonfile', default='bbox-val.json', help='submit_file_name', type=str)
    args = parser.parse_args()
    return args

underwater_classes = ['holothurian', 'echinus', 'scallop', 'starfish']

def post_predictions(predictions, img_shape):
    bboxes_list, scores_list, labels_list = [], [], []
    for i, bboxes in enumerate(predictions):
        if len(bboxes) > 0:
            detect_label = i
            for bbox in bboxes:
                xmin, ymin, xmax, ymax, score = bbox.tolist()

                xmin /= img_shape[1]
                ymin /= img_shape[0]
                xmax /= img_shape[1]
                ymax /= img_shape[0]
                bboxes_list.append([xmin, ymin, xmax, ymax])
                scores_list.append(score)
                labels_list.append(detect_label)

    return bboxes_list, scores_list, labels_list
    
def main():
    args = parse_args()
    config_file1 = './swa/swa_cascade_rcnn_r50_rfp_sac_iou_alldata-v3_e15/swa_cascade_rcnn_r50_rfp_sac_iou_alldata-v3_e15.py'  #
    checkpoint_file1 = './swa/swa_cascade_rcnn_r50_rfp_sac_iou_alldata-v3_e15/swa_model_12.pth'  
    config_file2 = './swa/cascade_rcnn_r50_rfp_sac_iou_ls_alldata-v3_e15.py'
    checkpoint_file2 = './swa/epoch_15.pth'
    
    device = 'cuda:0'
    cfg1 = Config.fromfile(config_file1)
    cfg2 = Config.fromfile(config_file2)
    # build model
    # model1
    model1 = build_detector(cfg1.model, test_cfg=cfg1.get('test_cfg'))
    load_checkpoint(model1, checkpoint_file1, map_location=device)
    # model2
    model2 = build_detector(cfg2.model, test_cfg=cfg2.get('test_cfg'))
    load_checkpoint(model2, checkpoint_file2, map_location=device)
    
    test_json_raw = json.load(open(cfg1.data.test.ann_file))
    imgid2name = {}
    for imageinfo in test_json_raw['images']:
        imgid = imageinfo['id']
        imgid2name[imageinfo['file_name']] = imgid
    wrap_fp16_model(model1)  # 采用fp16加速预测
    wrap_fp16_model(model2)
    
    # build the dataloader
    samples_per_gpu = cfg1.data.test.pop('samples_per_gpu', 1)  # aug_test不支持batch_size>1
    dataset = build_dataset(cfg1.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=4,
        dist=False,
        shuffle=False)
    model1 = MMDataParallel(model1, device_ids=[0])  # 为啥加？(不加就错了)
    model2 = MMDataParallel(model2, device_ids=[0])
    model1.eval()
    model2.eval()

    json_results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result1 = model1(return_loss=False, rescale=True, **data)
            result2 = model2(return_loss=False, rescale=True, **data)
        batch_size = len(result1)
        assert len(result1) == len(result2)
        
        result1 = result1[0]  # 每次只输入一张
        result2 = result2[0]
        img_metas = data['img_metas'][0].data[0]
        img_shape = img_metas[0]['ori_shape']
        bboxes, scores, labels = post_predictions(result1, img_shape)
        e_bboxes, e_scores, e_labels = post_predictions(result2, img_shape)
        bboxes_list = [bboxes, e_bboxes]
        scores_list = [scores, e_scores]
        labels_list = [labels, e_labels]
        bboxes, scores, labels = weighted_boxes_fusion(
            bboxes_list,
            scores_list,
            labels_list,
            weights=[1, 1],
            iou_thr=0.6,
            skip_box_thr=0.0001,
            conf_type='max')
        # basename = img_metas[0]['ori_filename']
        # image = cv2.imread(os.path.join(cfg.data.test.img_prefix, basename))
        for (box, score, label) in zip(bboxes, scores, labels):
            xmin, ymin, xmax, ymax = box.tolist()
            xmin, ymin, xmax, ymax = round(
                float(xmin) * img_shape[1],
                2), round(float(ymin) * img_shape[0],
                          2), round(float(xmax) * img_shape[1],
                                    2), round(float(ymax) * img_shape[0], 2)
            data = dict()
            data['image_id'] = imgid2name[img_metas[0]['ori_filename']]
            data['bbox'] = [xmin, ymin, xmax-xmin, ymax-ymin]
            data['score'] = float(score)
            data['category_id'] = label+1
            json_results.append(data)
        for _ in range(batch_size):
            prog_bar.update()
    mmcv.dump(json_results, args.jsonfile)
#                     x1, y1, x2, y2, score = bbox.tolist()
#                     if score >= 0.001:
#                         cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
#                         cv2.putText(image, underwater_classes[i] + ' ' + str(round(score, 2)),
#                                 (int(x1), int(y1 - 2)),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), thickness=2
#                                 )
#         cv2.imwrite(os.path.join('val_img', basename), image)
#         exit()

if __name__ == "__main__":
    main()