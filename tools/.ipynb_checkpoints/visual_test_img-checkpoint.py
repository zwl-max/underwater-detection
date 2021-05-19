import time
import os, cv2
import json
import mmcv
import glob 
from mmdet.apis import init_detector, show_result_pyplot, adaptive_inference_detector

underwater_classes = ['holothurian', 'echinus', 'scallop', 'starfish']
def main():
    config_file = './revised_dirs/cascade_rcnn_r50_rfp_sac_e15/cascade_rcnn_r50_rfp_sac_e15.py'  #
    checkpoint_file = './revised_dirs/cascade_rcnn_r50_rfp_sac_e15/epoch_15.pth'  
    test_path = '../data/test-A-image'  # test的图片所在的路径下
    model = init_detector(config_file, checkpoint_file, device='cuda')
    result = []
    for i, img_name in enumerate(glob.glob(test_path + '/*')):
        # img_name = '../data/test-A-image/000779.jpg'
        img_name = '../data/train/image/u004238.jpg'
        # print(img_name)
        image = cv2.imread(img_name)
        basename = os.path.basename(img_name)
        predict = adaptive_inference_detector(model, image)
        for i, bboxes in enumerate(predict):
            if len(bboxes) > 0:
#                 if i == 1 or i == 3:
#                     thre_score = 0.1
#                 else:
#                     thre_score = 0.05
                thre_score=0.001
                defect_label = underwater_classes[i]
                for bbox in bboxes:               
                    x1, y1, x2, y2, score = bbox.tolist()
                    if score >= thre_score:
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
                        cv2.putText(image, str(defect_label) + ' ' + str(round(score, 2)),
                                (int(x1), int(y1 - 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), thickness=2
                                )
        cv2.imwrite(os.path.join('val_img', basename), image)
        exit()

if __name__ == "__main__":
    main()