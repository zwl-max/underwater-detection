# underwater-object-detection-mmdetection


## [和鲸社区Kesci 水下目标检测算法赛（光学图像赛项）](https://www.heywhale.com/home/competition/605ab78821e3f6003b56a7d8/content)

### A榜：第9  0.57286428 
### B榜：第3  0.57534568 
### 单模 TTA(三尺度+水平翻转)
### 注：没有标注的图片不加入训练
### Baseline
1. cascade_rcnn_x101_64x4d_fpn_dcn_e15  
    + Backbone:
        + ResNeXt101-64x4d
    + Neck:
        + FPN
    + DCN
    + Global context(GC)
    + MS [(4096, 600), (4096, 1000)]
    + RandomRotate90°
    + 15epochs + step:[11, 13]  
    + A榜：0.55040585 
        + 注：不是所有数据
   
 
2. 基于1训练好的模型对训练数据进行清洗(tools/data_process/data_clean.py)
    + 1. 如果某张图片上所有预测框的confidence没有一个是大于0.9， 那么去掉该图片(即看不清的图片)
    + 2. 修正错误标注
        + 1. 先过滤掉confidence<0.1的predict boxes, 然后同GT boxes求iou
        + 2. 如果predict box同GT的最大iou大于0.6，但类别不一致， 那么就修正该gt box的类别


3. 基于2修正后的数据进行训练
   模型采用cascade_rcnn_r50_rfp_sac
    + Backbone:
        + ResNet50
    + Neck:
        RFP-SAC
    + GC + MS + RandomRotate90°
    + cascade_iou调整为：（0.55， 0.65， 0.75）
    + A榜： 0.56339531
        + 注：所有数据


4. 基于3训练好的模型进一步清洗数据
    + 模型同3： 
    + A榜：0.56945031
        + 注：所有数据
    

5. 基于4
    + SWA(12epochs)(https://arxiv.org/abs/2012.12645)
    + A榜：0.57286428
        + 注：所有数据
    

## 项目运行的资源环境
+ 操作系统：Ubuntu 18.04.2
+ GPU：1块2080Ti
+ Python：Python 3.7.7
+ NVIDIA依赖：
    - NVCC: Cuda compilation tools, release 10.1, V10.1.243
    - CuDNN 7.6.5
+ 深度学习框架：
    - PyTorch: 1.8.1
    - TorchVision: 0.9.1
    - OpenCV: 4.5.1
    - MMCV: 1.2.4+cu101
    - MMCV Compiler: GCC 7.5
    - MMCV CUDA Compiler: 10.1
    - MMDetection: 2.10.0+0489bcb

## 环境安装及编译
+ mmdetection安装
    - 参考[mmdetection](https://github.com/open-mmlab/mmdetection)


## 预训练模型下载
 - 下载mmdetection官方开源的htc的[resnext 64×4d 预训练模型](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth)
 - 下载mmdetection官方开源的htc的[DetectoRS | HTC + ResNet-50 预训练模型](http://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r50_1x_coco/detectors_htc_r50_1x_coco-329b1453.pth) 


## 模型训练与预测
  - **训练**

	1. 运行：
       
       r50_rfp_sac (htc pretrained):
          + python tools/train.py configs/cascade_rcnn_r50_rfp_sac_iou_e15_alldata_v3.py --no-validate
        
    2. SWA -- 训练12轮
       
       + python tools/train.py configs/swa_cascade_rcnn_r50_rfp_sac_iou_alldata-v3_e15.py --no-validate

  - **预测**

    1. 运行: 
       + python tools/test.py configs/swa_cascade_rcnn_r50_rfp_sac_iou_alldata-v3_e15.py  ./swa/swa_cascade_rcnn_r50_rfp_sac_iou_alldata-v3_e15/swa_model_12.pth  --format-only  --cfg-options "jsonfile_prefix=./submit"
         + 注： 采用fp16加速(配置文件添加fp16 = dict(loss_scale=512.0))
        
    2. 预测结果文件名submit.bbox.json

    3. 转化mmd预测结果为提交csv格式文件：
       
       python tools/post_process/json2submit.py --test_json submit.bbox.json --submit_file submit.csv

       最终符合官方要求格式的提交文件 submit.csv 位于 submit目录下
    

## 尝试过但不work：

+ 数据增强
  + 竖直翻转和对角翻转
    + dict(type='RandomFlip', direction='vertical', flip_ratio=0.5), 
    + dict(type='RandomFlip', direction='diagonal', flip_ratio=0.5),
  + mixup
  + 引入往年数据
    + 通过md5去掉重复的图片
  + autoaugment(v2, v4)
  + mosaic
  
+ 模型部分
  + 对特征进行roimix(可能是代码问题)[RoIMix](https://arxiv.org/abs/1911.03029)
  + rpn采样器换成ATSS
  + PAFPN
  + DeformRoIPool替换RoIAlign
  + bbox_head: Shared4Conv1FCBBoxHead替换Shared2FCBBoxHead
  + ResNeSt101
  + Res2Net101
    
## Reference 感谢大佬的开源
   - [郑烨-underwater-detection](https://github.com/zhengye1995/underwater-object-detection)
   - [郑烨-tile-detection](https://github.com/zhengye1995/Tianchi-2021-Guangdong-Tile-Detection)
   - [milleniums](https://github.com/milleniums/underwater-object-detection-mmdetection)
   - [Wakinguup-underwater-detection](https://github.com/Wakinguup/Underwater_detection)
   - [clw5180-tile-detection](https://github.com/clw5180/mmdetection_clw)
   - [tile-detection](https://github.com/MySuperSoul/TileDetection)


## Contact
    author: tricks
    qq：1227125939
    email: 17760867927@163.com
# underwater-detection
