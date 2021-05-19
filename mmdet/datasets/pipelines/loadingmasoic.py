import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES

import copy
import random
import cv2
import os

@PIPELINES.register_module()
class LoadMosaicImageAndAnnotations(object):
    """Load an image from file.
    https://github.com/clw5180/mmdetection_clw/blob/main/mmdet/datasets/pipelines/loading.py

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self, 
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 hsv_aug=True,
                 h_gain=0.014,
                 s_gain=0.68,
                 v_gain=0.36,
                 skip_box_w=0,
                 skip_box_h=0,
                 image_shape=[1024, 1024]):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain
        self.hsv_aug = hsv_aug
        self.skip_box_w = skip_box_w
        self.skip_box_h = skip_box_h
        self.image_shape = image_shape

    def __call__(self, results):
        if len(results) == 1:
            results = self._load_image_annotations(results, 0)
        elif len(results) == 4:
            results = self._load_mosaic_image_and_annotations(results)
        else:
            assert False
        return results

    def _crop_patch(self, img, gt_bboxes, gt_labels, crop_size):

        H, W, C = img.shape
        px, py = crop_size  # 要切的patch的尺寸

        if px > W or py > H:  # 如果要切的patch比图像本身还大, 那就只能pad原图了, 见下；
            pad_w, pad_h = 0, 0

            if px > W:
                pad_w = px - W
                W = px
            if py > H:
                pad_h = py - H
                H = py

            img = cv2.copyMakeBorder(img, 0, int(pad_h), 0, int(pad_w), cv2.BORDER_CONSTANT, 0)  # top, bottom, left, right ：相应方向上的边框宽度

        obj_num = gt_bboxes.shape[0]
        select_gt_id = np.random.randint(0, obj_num)  # 随机选择一个gt，准备在它四周切下一个patch
        x1, y1, x2, y2 = gt_bboxes[select_gt_id, :]  # 选的某个gt，拿到它的 xyxy

        if x2 - x1 > px:  # clw note: 如果bbox的宽度比要切的patch的宽度还宽
            nx = np.random.randint(x1, x2 - px + 1)  # 就切一部分
        else:  # clw note: 一般都是进这里
            nx = np.random.randint(max(x2 - px, 0),
                                   min(x1 + 1, W - px + 1))

        if y2 - y1 > py:
            ny = np.random.randint(y1, y2 - py + 1)
        else:
            ny = np.random.randint(max(y2 - py, 0), min(y1 + 1, H - py + 1))

        patch_coord = np.zeros((1, 4), dtype="int")
        patch_coord[0, 0] = nx
        patch_coord[0, 1] = ny
        patch_coord[0, 2] = nx + px
        patch_coord[0, 3] = ny + py

        # index = self._compute_overlap(patch_coord, gt_bboxes, 0.5)
        index = self._compute_overlap(patch_coord, gt_bboxes, 0.7)  # clw modify
        index = np.squeeze(index, axis=0)
        index[select_gt_id] = True  # 这里貌似写不写TRue都行，因为上面计算overlap，patch和内部的gt的overlap一定是1

        patch = img[ny: ny + py, nx: nx + px, :]
        gt_bboxes = gt_bboxes[index, :]
        gt_labels = gt_labels[index]

        gt_bboxes[:, 0] = np.maximum(gt_bboxes[:, 0] - patch_coord[0, 0], 0)  # 如果patch左边缘在gt左边缘的右边，那么这里就会算出负数，所以限制到0
        gt_bboxes[:, 1] = np.maximum(gt_bboxes[:, 1] - patch_coord[0, 1], 0)
        gt_bboxes[:, 2] = np.minimum(gt_bboxes[:, 2] - patch_coord[0, 0],
                                     px - 1)  # 如果patch右边缘在gt右边缘左边，那么这里算出来的值就会比patch宽度还大，因此要限制到patch宽度
        gt_bboxes[:, 3] = np.minimum(gt_bboxes[:, 3] - patch_coord[0, 1], py - 1)

        return patch, gt_bboxes, gt_labels

    def _compute_overlap(self, a, b, over_threshold=0.5):
        """
        Parameters
        ----------
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float
        Returns
        -------
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
        ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

        iw = np.maximum(iw, 0)
        ih = np.maximum(ih, 0)

        # ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
        ua = area

        ua = np.maximum(ua, np.finfo(float).eps)

        intersection = iw * ih

        overlap = intersection / ua
        index = overlap > over_threshold
        return index

    def _load_mosaic_image_and_annotations(self, results):
        indexes = [0, 1, 2, 3]
        result_boxes = []
        result_labels = []  # clw note: TODO !!!!!
        results_c = copy.deepcopy(results)  # [ result, result, result, result ]  result: dict, 'img_info', 'ann_info', ....
        results = results[0]
        imsize = self.image_shape[0]
        #print(imsize)
        w, h = self.image_shape[0], self.image_shape[1]
        s = imsize // 2
        #xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        xc, yc = [int(random.uniform(imsize * 0.4, imsize * 0.6)) for _ in range(2)]  # clw modify
        # result_image = np.full((imsize, imsize, 3), 0, dtype=np.float32)  # large image, will be composed by 4 small images
        result_image = np.full((imsize, imsize, 3), 0, dtype=np.uint8)  
        for i, index in enumerate(indexes):
            result = self._load_image_annotations(results_c, index)
            img = result['img']
            # img = result['img'].astype(np.uint8)
            #print(result.keys())
            hs, ws = img.shape[:2]  # clw note: small image

            gt_bboxes = result['gt_bboxes']
            gt_labels = result['gt_labels']
            # result_masks.append(result['gt_masks'])

            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - ws, 0), max(yc - hs, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - hs, 0), min(xc + ws, s * 2), yc
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - ws, 0), yc, xc, min(s * 2, yc + hs)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + ws, s * 2), min(s * 2, yc + hs)

            img_cropped, gt_bboxes_cropped, gt_labels_cropped = self._crop_patch(img, gt_bboxes, gt_labels, (x2a-x1a, y2a-y1a))
            result_image[y1a:y2a, x1a:x2a, :] = img_cropped

            # x1b = int(random.uniform(0, w - (x2a - x1a)))
            # y1b = int(random.uniform(0, h - (y2a - y1a)))
            # padw = x1a - x1b
            # padh = y1a - y1b
            # x2b = x2a - padw
            # y2b = y2a - padh
            # result_image[y1a:y2a, x1a:x2a, :] = image[y1b:y2b, x1b:x2b, :]
            # print('result img:', result_image.shape, y1a,y2a, x1a,x2a, 'cut img:',  image.shape, y1b,y2b,  x1b,x2b) # for debug


            # if i == 0:
            #     pass
            # elif i == 1:  # top right
            #     #gt_bboxes_cropped[:, 0::2] += xc
            #     gt_bboxes_cropped[:, 0::2] += x1a
            # elif i == 2:  # bottom left
            #     #gt_bboxes_cropped[:, 1::2] += yc
            #     gt_bboxes_cropped[:, 1::2] += y1a
            # elif i == 3:  # bottom right
            #     #gt_bboxes_cropped[:, 0::2] += xc
            #     gt_bboxes_cropped[:, 0::2] += x1a
            #     #gt_bboxes_cropped[:, 1::2] += yc
            #     gt_bboxes_cropped[:, 1::2] += y1a
            gt_bboxes_cropped[:, 0::2] += x1a
            gt_bboxes_cropped[:, 1::2] += y1a

            # clean bbox out of region   TODO: this version is coarse
            # mask = (boxes[:, 2] > x1a) & (boxes[:, 0] < x2a) & (boxes[:, 3] > y1a) & (boxes[:, 1] < y2a)  # filter the gt which is out of the cropped region of origin image
            # boxes = boxes[mask]

            if gt_bboxes_cropped.size == 0:
                # boxes = np.zeros((0, 4), dtype=np.float32)
                # result_boxes.append(boxes)
                continue
            else:
                # np.clip(boxes[:, 0::2], x1a, x2a, out=boxes[:, 0::2])  # clw modify: the same with up
                # np.clip(boxes[:, 1::2], y1a, y2a, out=boxes[:, 1::2])
                # mask2 = (boxes[:, 2] - boxes[:, 0] > self.skip_box_w) & (boxes[:, 3] - boxes[:, 1] > self.skip_box_h) # filter the cropped gt which is too thin, which may lead to 'RuntimeError: CUDA error: device-side assert triggered' when build_target() if using hflip data_augment(index 13-0, 26-0, 52-0 cause this problem)
                # boxes = boxes[mask2]
                result_boxes.append(gt_bboxes_cropped)
                result_labels.append(gt_labels_cropped)

        result_boxes = np.concatenate(result_boxes, 0)
        result_labels = np.concatenate(result_labels, 0)
        # results = self._load_image_boxes(results, 0)

        ## 可视化确认结果无误
        # filename = results['img_info']['file_name']
        # import cv2
        # img_out = result_image.copy()
        # for box in result_boxes:
        #     cv2.rectangle(img_out, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
        # cv2.imwrite('/home/featurize/work/underwater-object-detection/image/' + filename, img_out)
        # if self.template_path is not None:
        #     cv2.imwrite('/home/user/' + filename.split('.')[0] + '_t.jpg', result_image_t)
        ##

        results['ann_info']['bboxes'] = result_boxes
        results['ann_info']['labels'] = result_labels

        # masks = []
        # for box in result_boxes:
        #     min_x = box[0]
        #     min_y = box[1]
        #     max_x = box[2]
        #     max_y = box[3]
        #     mask_h = max_y - min_y
        #     mask_w = max_x - min_x
        #     masks.append([[min_x, min_y, min_x, min_y + 0.5 * mask_h, min_x, max_y, min_x + 0.5 * mask_w, max_y, max_x,
        #                    max_y, max_x,
        #                    max_y - 0.5 * mask_h, max_x, min_y, max_x - 0.5 * mask_w, min_y]])
        #
        # results['ann_info']['masks'] = masks

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'], results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        #result_image = img_as_ubyte(result_image / 255.0)  # clw delete
        if self.hsv_aug:
            augment_hsv(img=result_image, hgain=self.h_gain, sgain=self.s_gain, vgain=self.v_gain)
        results['img'] = result_image
        results['img_fields'] = ['img']
        results['img_shape'] = result_image.shape
        results['ori_shape'] = result_image.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = result_image.shape
        #results['scale_factor'] = 1.0   # clw delete: if so, the Resize() after this is invalid
        num_channels = 1 if len(result_image.shape) < 3 else result_image.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def _load_image_annotations(self, results, k):
        results = results[k]
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)

        if self.to_float32:
            img = img.astype(np.float32)
        
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        """Private function to load label annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        """Private function to convert masks represented with polygon to
        bitmaps.
        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.
        Returns:
            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.
        Args:
            polygons (list[list]): Polygons of one instance.
        Returns:
            list[numpy.ndarray]: Processed polygons.
        """

        polygons = [np.array(p) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):
        """Private function to load mask annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """

        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        """Private function to load semantic segmentation annotations.
        Args:
            results (dict): Result dict from :obj:`dataset`.
        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        results['gt_semantic_seg'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'image_shape={self.image_shape}, '
        repr_str += f'hsv_aug={self.hsv_aug}, '
        repr_str += f'h_gain={self.h_gain}, '
        repr_str += f's_gain={self.s_gain}, '
        repr_str += f'v_gain={self.v_gain}, '
        repr_str += f'skip_box_w={self.skip_box_w}, '
        repr_str += f'skip_box_h={self.skip_box_h}, '
        repr_str += f'to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}', "
        repr_str += f'with_seg={self.with_seg})'
        repr_str += f'poly2mask={self.poly2mask})'
        repr_str += f'poly2mask={self.file_client_args})'
        repr_str += f'file_client_args={self.file_client_args})'
        return repr_str
    
def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8
    
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)