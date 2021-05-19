import time, argparse
from tqdm import tqdm
import os, cv2
import json
import mmcv
import glob 
import torch
from mmdet.apis import init_detector, show_result_pyplot, adaptive_inference_detector, inference_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmcv.runner import wrap_fp16_model, load_checkpoint
from mmcv.cnn import fuse_conv_bn
from mmcv import Config
from mmcv.parallel import MMDataParallel

def parse_args():
    parser = argparse.ArgumentParser(description='json2submit_nms')
    parser.add_argument('--jsonfile', default='bbox-val.json', help='submit_file_name', type=str)
    args = parser.parse_args()
    return args

underwater_classes = ['holothurian', 'echinus', 'scallop', 'starfish']

def main():
    args = parse_args() 
    config_file = './cascade_rcnn_r50_rfp_sac_iou_ls_alldata-v3_e15.py'
    checkpoint_file = 'epoch_15.pth'
    device = 'cuda:0'
    cfg = Config.fromfile(config_file)
    # build model
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_file, map_location=device)
    test_json_raw = json.load(open(cfg.data.test.ann_file))
    imgid2name = {}
    for imageinfo in test_json_raw['images']:
        imgid = imageinfo['id']
        imgid2name[imageinfo['file_name']] = imgid
        # imgid2name[imgid] = imageinfo['file_name']
    wrap_fp16_model(model)  # 采用fp16加速预测
    # model = fuse_conv_bn(model)  # 加上后出错
    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)  # aug_test不支持batch_size>1
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    model = MMDataParallel(model, device_ids=[0])  # 为啥加？(不加就错了)
    model.eval()
    json_results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        batch_size = len(result)
        result = result[0]  # 每次只输入一张
        img_metas = data['img_metas'][0].data[0]
        # print(result)
        # predict = adaptive_inference_detector(model, image)
        # basename = img_metas[0]['ori_filename']
        # image = cv2.imread(os.path.join(cfg.data.test.img_prefix, basename))
        for i, bboxes in enumerate(result):
            if len(bboxes) > 0:
                for bbox in bboxes:
                    x1, y1, x2, y2, score = bbox.tolist()
                    if score >= 0.001:
                        data = dict()
                        data['image_id'] = imgid2name[img_metas[0]['ori_filename']]
                        data['bbox'] = [x1, y1, x2-x1, y2-y1]
                        data['score'] = float(score)
                        data['category_id'] = i+1
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