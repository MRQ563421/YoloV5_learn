```python
detect.py 预测

def parse_opt():
    parser = argparse.ArgumentParser()
    
    # 从网络下载模型
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    
    #要训练的模型位于哪里，可以是文件夹，视频链接等
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    
    # 修改网络尺寸，训练时将图片等比例缩小，输出时在放大到原来
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    
    # 预测概率高于0.25显示，否则不显示
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    
    # iou 全名Intersection over Union用来去除差不多概率的锚框
    # 多个概率差不多的框圈住同一个目标的时候使用，只有IOU大于阈值0.45的时候才对多个框图进行处理，否则不处理
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    
    
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    
    #运行设备
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    # 是否在运行时显示结果，如果设置该参数就就True，显示结果，不设置默认为FALSE，不显示结果，action表示只要设置有参数就直接是ture     可以直接在pycharm上面的Edit Configuration 中填写--view-img保证在运行时实时显示，而不用在控制台输入参数
    parser.add_argument('--view-img', action='store_true', help='show results')
    
    
    # 是否保存成text格式的文件 有参数设置为ture 也可直接在Edit Configuration中的parameters中填写
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    
    # 不要保存为图片或者视频
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    
    # nargs='+'表示这个参数可以接受多个赋值  --classes 0 只显示0号类别 --classes 0 2 3'显示0 2 3号类别其他不显示
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    
    # 增强检测效果
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
     # 增强检测效果
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    
    
    
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    
    # 保留模型中重要的那一部分，一些不重要的参数（优化器什么的）可以忽视
    parser.add_argument('--update', action='store_true', help='update all models')
    
    #默认的结果保存的地方
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    #保存结果的名字
    parser.add_argument('--name', default='exp', help='save results to project/name')
    
    # 设置为True则将结果保存在name指定的文件夹（如exp）当中，不再新创建文件夹(exp1\exp2\exp3\exp4)
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    
    
    #将所有参数放到 opt当中
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt
```

# 模型预测detect.py









# 训练 train.py

## 参数

```python
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
     
    # dafault设置为空，则使用程序自己训练模型，有参数的话，列如'yolov5s.pt'则会使用已经训练好模型 训练\初始化，自己搞的话可以设置为空   default=ROOT / 'yolov5s.pt'这里可以填写yolov5s,pt/yolov5m.pt/yolov5l.pt/yolov5x.pt
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    
    #模型结构地址，表明模型的结构是什么  可以参考yolov5m.yaml，表明是这个模型的结构是什么，可以这样设置：default='C:\Users\MRQ\Desktop\DeepLearning\yolov5-6.0\models\yolov5m.yaml'  ， default里面参数是路径名
    #可以参考后面的yolov5m.yaml结构
    
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    
    # 指明要训练的数据集在哪里， default=‘’里面填写数据集的路径，可以参考下面的coco128.yaml
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    
    
    
    
    # 超参数 https://blog.csdn.net/xu_fu_yong/article/details/96102999?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168239139216800188528742%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168239139216800188528742&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-96102999-null-null.142^v86^insert_down1,239^v2^insert_chatgpt&utm_term=%E8%B6%85%E5%8F%82%E6%95%B0&spm=1018.2226.3001.4187讲解
    # 对网络进行微调 什么是超参数，参数和超参数的区别？
    #区分两者最大的一点就是是否通过数据来进行调整，模型参数通常是有数据来驱动调整，超参数则不需要数据来驱动，而是在训练前或者训练中人为的进行调整的参数。例如卷积核的具体核参数就是指模型参数，这是有数据驱动的。而学习率则是人为来进行调整的超参数。这里需要注意的是，通常情况下卷积核数量、卷积核尺寸这些也是超参数，注意与卷积核的核参数区分。
	#通常可以将超参数分为三类：网络参数、优化参数、正则化参数
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    
    
    #训练的轮数
    parser.add_argument('--epochs', type=int, default=300)
    
    # 从数据中拿多少数据送到网络当中  batch_size
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    
    # 训练集和测试集的大小
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    
    # 将图片填充，不足的填充上去，现在的填充比较少，可以加快训练速度，见--rect，在下面
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    
    # 从已经训练过的模型开始继续训练，接着上一步开始训练，default=后面可以设置路径，告诉我们从哪里开始训练，这里的路径需要是当初训练的时候模型保存的位置，因为需要相关的配置信息
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    
    # 是否只保存最后一个模型
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    #是否只进行最后一次测试
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    
    #是否采用锚点，本参数默认开启，我们也需要他开启
    #锚框的作用：在进行目标检测任务的基本思路：通过设定众多的候选框，然后针对候选框进行分类和微调，找到目标中最接近的真实框，实现目标检测。这里的候选框也就是锚框
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    
    
    
    # 训练时候的默认进程数，建议设置为0
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', action='store_true', help='W&B: Upload dataset as artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt
```



## yolo v5m.yaml

![image-20230425105035990](C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230425105035990.png)



```python
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
nc: 80  # number of classes   这个模型有多少个类别
depth_multiple: 0.67  # model depth multiple  
width_multiple: 0.75  # layer channel multiple  分层通道倍数
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```







## coco128.yaml

![image-20230425105106821](C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230425105106821.png)



```python
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# COCO128 dataset https://www.kaggle.com/ultralytics/coco128 (first 128 images from COCO train2017)
# Example usage: python train.py --data coco128.yaml
# parent
# ├── yolov5
# └── datasets
#     └── coco128  ← downloads here


# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco128  # dataset root dir 这个数据集
train: images/train2017  # train images (relative to 'path') 128 images
val: images/train2017  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes
nc: 80  # number of classes 数据集的类别，每个类别叫啥
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']  # class names


# Download script/URL (optional) 在哪里下载这个数据集
download: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip
```



## 超参数hyp.scratch-low.yaml

![image-20230425105934078](C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230425105934078.png)



```python
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Hyperparameters for low-augmentation COCO training from scratch
# python train.py --batch 64 --cfg yolov5n6.yaml --weights '' --data coco.yaml --img 640 --epochs 300 --linear
# See tutorials for hyperparameter evolution https://github.com/ultralytics/yolov5#tutorials

lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)  初始学习速率
lrf: 0.01  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4
warmup_epochs: 3.0  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
box: 0.05  # box loss gain
cls: 0.5  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 1.0  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.20  # IoU training threshold
anchor_t: 4.0  # anchor-multiple threshold
# anchors: 3  # anchors per output layer (0 to ignore)
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.1  # image translation (+/- fraction)
scale: 0.5  # image scale (+/- gain)
shear: 0.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.0  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 1.0  # image mosaic (probability)
mixup: 0.0  # image mixup (probability)
copy_paste: 0.0  # segment copy-paste (probability)
```



## --rect rectangular training

![image-20230425110734812](C:\Users\MRQ\AppData\Roaming\Typora\typora-user-images\image-20230425110734812.png)

第一个就是将不是矩阵的填充成矩阵，第二个不用完全填充成矩阵