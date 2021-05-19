import json
import numpy as np
import os
from tqdm import tqdm
import mmcv
from PIL import Image

underwater_classes = ['holothurian', 'echinus', 'scallop', 'starfish']
from mmdet.core.visualization import imshow_det_bboxes

def bbox_iou(box1, box2):
    """
    Calculate the IOU between box1 and box2.

    :param boxes: 2-d array, shape(n, 4)
    :param anchors: 2-d array, shape(k, 4)
    :return: 2-d array, shape(n, k)
    """
    # Calculate the intersection,
    # the new dimension are added to construct shape (n, 1) and shape (1, k),
    # so we can get (n, k) shape result by numpy broadcast
    box1 = box1[:, np.newaxis]  # [n, 1, 4]
    box2 = box2[np.newaxis]     # [1, k, 4]

    xx1 = np.maximum(box1[:, :, 0], box2[:, :, 0])
    yy1 = np.maximum(box1[:, :, 1], box2[:, :, 1])
    xx2 = np.minimum(box1[:, :, 2], box2[:, :, 2])
    yy2 = np.minimum(box1[:, :, 3], box2[:, :, 3])
    w = np.maximum(0, xx2-xx1+1)
    h = np.maximum(0, yy2-yy1+1)
    inter = w * h
    area1 = (box1[:, :, 2] - box1[:, :, 0] + 1) * (box1[:, :, 3] - box1[:, :, 1] + 1)
    area2 = (box2[:, :, 2] - box2[:, :, 0] + 1) * (box2[:, :, 3] - box2[:, :, 1] + 1)
    ious = inter / (area1 + area2 - inter)

    return ious

label_ids = {name: i + 1 for i, name in enumerate(underwater_classes)}
def get_segmentation(points):

    return [points[0], points[1], points[2] + points[0], points[1],
             points[2] + points[0], points[3] + points[1], points[0], points[3] + points[1]]

def generate_json(img_root, annos, out_file):
    images = []
    annotations = []

    img_id = 1
    anno_id = 1
    for anno in tqdm(annos):
        img_name = anno[0]['image']
        img_path = os.path.join(img_root, img_name)
        w, h = Image.open(img_path).size
        img = {"file_name": img_name, "height": int(h), "width": int(w), "id": img_id}
        images.append(img)

        annotation = []
        for img_anno in anno:
            category_id = img_anno['category_id']
            xmin, ymin, w, h = img_anno['bbox']
            area = w * h
            segmentation = get_segmentation([xmin, ymin, w, h])
            annotation.append({
                "segmentation": segmentation,
                "area": area,
                "iscrowd": 0,
                "image_id": img_id,
                "bbox": [xmin, ymin, w, h],
                "category_id": category_id,
                "id": anno_id,
                "ignore": 0})
            anno_id += 1
        annotations.extend(annotation)
        img_id += 1
    categories = []
    for k, v in label_ids.items():
        categories.append({"name": k, "id": v})
    final_result = {"images": images, "annotations": annotations, "categories": categories}
    mmcv.dump(final_result, out_file)


if __name__ == '__main__':
    np.random.seed(121)
    data_json_raw = json.load(open("../underwater_data/train/annotations/trainall-revised-v3.json", "r"))  # gt box
    data_json = json.load(open("../underwater_data/train/annotations/revisedv3.bbox.json", "r"))  # pred box
    img = data_json_raw['images']

    unclear_anno_img = []  # 看不清的图片，自己记录

    images = []
    gt_imgid2anno = {}    # 真实图像的box
    pred_imgid2anno = {}  # 预测图像的box
    imgid2name = {}       # 图像名
    for imageinfo in data_json_raw['images']:  # 真实标注的image name
        imgid = imageinfo['id']
        imgid2name[imgid] = imageinfo['file_name']
    # print(len(imgid2name))  # 7600

    for anno in data_json_raw['annotations']:  # 真实标签
        img_id = anno['image_id']
        if img_id not in gt_imgid2anno:
            gt_imgid2anno[img_id] = []
        gt_imgid2anno[img_id].append(anno)

    for anno in data_json:  # 预测标签
        img_id = anno['image_id']
        if img_id not in pred_imgid2anno:
            pred_imgid2anno[img_id] = []
        pred_imgid2anno[img_id].append(anno)

    revised_annos = []
    for imgid in tqdm(gt_imgid2anno.keys()):
        if imgid2name[imgid] in unclear_anno_img:  # 看不清的图像，不加入训练
            # print(imgid2name[imgid])
            continue
        annos = pred_imgid2anno[imgid]
        pred_boxes = []
        for anno in annos:
            xmin, ymin, w, h = anno['bbox']
            xmax = xmin + w
            ymax = ymin + h
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            confidence = anno['score']
            class_id = int(anno['category_id']) - 1
            pred_boxes.append([xmin, ymin, xmax, ymax, confidence, class_id])
        pred_boxes = np.array(pred_boxes)
        pred_boxes = pred_boxes[pred_boxes[:, 4] > 0.1]  # 过滤掉低score

        gt_boxes = []
        revised_gt = []
        if imgid in gt_imgid2anno.keys():
            for anno in gt_imgid2anno[imgid]:
                xmin, ymin, w, h = anno['bbox']
                xmax = xmin + w
                ymax = ymin + h
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                class_id = int(anno['category_id']) - 1
                gt_boxes.append([xmin, ymin, xmax, ymax, class_id])
            gt_boxes = np.array(gt_boxes)

        if len(pred_boxes) == 0:  # 当前img没有预测框
            # 不用修正gt box的类别
            for anno in gt_imgid2anno[imgid]:
                revised_gt.append({'image':imgid2name[imgid],
                                   'bbox':anno['bbox'], 'category_id':anno['category_id']})
            revised_annos.append(revised_gt)
            # filename = os.path.join('../underwater_data/train/image', imgid2name[imgid])
            # img = cv2.imread(filename)
            # basename = os.path.basename(filename)
            # imshow_det_bboxes(img, gt_boxes[:, :4], gt_boxes[:, 4], class_names=underwater_classes,
            #                   show=False,
            #                   out_file=os.path.join('../underwater_data/train/no_pred_box-0.95/' + basename))
            continue

        ious = bbox_iou(pred_boxes[:, :4], gt_boxes[:, :4])  # [n, k]
        max_idx = np.argmax(ious, axis=0)  # [k,]
        max_value = np.amax(ious, axis=0)  # [k,]
        pred_boxes = pred_boxes[max_idx]   # gt 对应的 pred box

        flag = False
        diff_boxes = []  # 可能标注错误的box
        for i in range(len(gt_boxes)):
            if gt_boxes[i][4] != int(pred_boxes[i][5]) and max_value[i] >= 0.6:
                # print(imgid2name[imgid])
                diff_boxes.append(pred_boxes[i])
                xmin, ymin, xmax, ymax = gt_boxes[i][:4]
                category_id = pred_boxes[i][5] + 1  # 修改此gt box的类别
                revised_gt.append({'image':imgid2name[imgid],
                                   'bbox':[xmin, ymin, xmax-xmin, ymax-ymin], 'category_id':category_id})
                flag = True
            else:
                xmin, ymin, xmax, ymax = gt_boxes[i][:4]
                category_id = gt_boxes[i][4] + 1  # 不用修改gt box的类别
                revised_gt.append({'image': imgid2name[imgid],
                                   'bbox': [xmin, ymin, xmax - xmin, ymax - ymin], 'category_id': category_id})
        revised_annos.append(revised_gt)

    print(len(revised_annos))
    # 划分数据集， 10%作为验证集
    np.random.shuffle(revised_annos)
    train_size = int(len(revised_annos) * 0.9)
    train_revised_annos = revised_annos[:train_size]
    val_revised_annos = revised_annos[train_size:]
    print("all revised data!")
    generate_json('../underwater_data/train/image', revised_annos,
                  '../underwater_data/train/annotations/trainall-revised-v4.json')
    print("train revised data!")
    generate_json('../underwater_data/train/image', train_revised_annos,
                  '../underwater_data/train/annotations/train-revised-v4.json')
    print("val revised data!")
    generate_json('../underwater_data/train/image', val_revised_annos,
                  '../underwater_data/train/annotations/val-revised-v4.json')
        # if flag:
        #     diff_boxes = np.array(diff_boxes)
        #     # print(imgid2name[imgid])
        #     filename = os.path.join('../underwater_data/train/image', imgid2name[imgid])
        #     img = cv2.imread(filename)
        #     basename = os.path.basename(filename)
            # imshow_det_bboxes(img, gt_boxes[:, :4], gt_boxes[:, 4], class_names=underwater_classes,
            #                   show=False,
            #                   out_file=os.path.join('../underwater_data/train/nosiy_image/' + basename))
            # imshow_det_bboxes(img, diff_boxes[:, :4], diff_boxes[:, 5].astype(np.int), class_names=underwater_classes,
            #                   bbox_color='red',
            #                   text_color='red',
            #                   show=False,
            #                   out_file=os.path.join('../underwater_data/train/nosiy_image-pred/' + basename))
            # imshow_det_bboxes(img, pred_boxes[:, :4], pred_boxes[:, 5].astype(np.int), class_names=underwater_classes,
            #                   bbox_color='red',
            #                   text_color='red',
            #                   show=False,
            #                   out_file=os.path.join('../underwater_data/train/image-pred/' + basename))


