import time, argparse
import asyncio, mmcv
import torch, json, os
from tqdm import tqdm
from mmdet.apis import init_detector, async_inference_detector, inference_detector
from mmdet.utils.contextmanagers import concurrent

def parse_args():
    parser = argparse.ArgumentParser(description='json2submit_nms')
    parser.add_argument('--jsonfile', default='bbox.json', help='submit_file_name', type=str)
    args = parser.parse_args()
    return args

underwater_classes = ['holothurian', 'echinus', 'scallop', 'starfish']

async def main():
    args = parse_args()
    config_file = './swa/swa_cascade_rcnn_r50_rfp_sac_iou_alldata-v3_e15/swa_cascade_rcnn_r50_rfp_sac_iou_alldata-v3_e15.py'  #
    checkpoint_file = './swa/swa_cascade_rcnn_r50_rfp_sac_iou_alldata-v3_e15/swa_model_12.pth'  
    device = 'cuda:0'
    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)
    
    test_json_raw = json.load(open("../data/train/annotations/testA.json", "r"))
    test_A_root = '../data/test-A-image'
    imgid2name = {}
    for imageinfo in test_json_raw['images']:
        imgid = imageinfo['id']
        imgid2name[imgid] = imageinfo['file_name']
        
    # queue is used for concurrent inference of multiple images
    streamqueue = asyncio.Queue()
    # queue size defines concurrency level
    streamqueue_size = 3

    for _ in range(streamqueue_size):
        streamqueue.put_nowait(torch.cuda.Stream(device=device))

    # test a single image and show the results
    # img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
    
    async with concurrent(streamqueue):
        json_results = []
        for imgid, img_name in tqdm(imgid2name.items()):
            # print(img_name)
            image = mmcv.imread(os.path.join(test_A_root, img_name))
            result = inference_detector(model, image)
            for i, bboxes in enumerate(result):
                if len(bboxes) > 0:
                    thre_score = 0.001
                    # defect_label = underwater_classes[i]
                    for bbox in bboxes:
                        x1, y1, x2, y2, score = bbox.tolist()
                        if score >= thre_score:
                            data = dict()
                            data['image_id'] = imgid
                            data['bbox'] = [x1, y1, x2-x1, y2-y1]
                            data['score'] = float(score)
                            data['category_id'] = i+1
                            json_results.append(data)

    mmcv.dump(json_results, args.jsonfile)

asyncio.run(main())