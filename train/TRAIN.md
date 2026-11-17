# 训练说明

这份文件记录板子上那个方块检测模型是怎么来的。数据集和权重都不进仓库（体积大），
这里只留能复现的过程。

## 环境

- Ultralytics 8.4.173
- Python 3.11.16 / torch 2.11.0+cu128
- 显卡 RTX 5060 Laptop 8G
- 基础模型 yolov8n.pt（Ultralytics 官方预训练权重）
- 类别：只有一类 box（方块）

## 第一步：造数据集

跑 build_box_dataset.py，把人工标的框和图片整理成 YOLO 能直接用的目录结构。

输入（本机原路径 D:\yolo_course\xiaosai_arena_202610\arena_box\ 下）：

- label\box_label.zip、box_label1.zip —— 人工标的 402 个框，class 0
- row_clean\box\*.jpg —— 抹掉黄字后的 402 张正样本
- row_clean\blanket\*.jpg —— 69 张负样本（配 0 字节同名 txt）

输出：

- arena_box_ds\images\{train,val,test}
- arena_box_ds\labels\{train,val,test}

划分规则（脚本里写死的）：

- 正样本按序号分段：000-299 进 train，300-349 进 val，350-401 进 test
- 负样本按排序后取模散开：i%9==0 进 test，i%9==4 进 val，其余进 train

脚本只复制文件，不动源目录；如果输出目录已存在会先清空。

## 第二步：训练

data.yaml 里的 path 已改成相对路径，指向 ../datasets/arena_box_ds。
把数据集放好之后，在仓库根目录执行：

yolo detect train model=yolov8n.pt data=train/data.yaml imgsz=480 epochs=100 batch=16 device=0 workers=0 seed=0 project=runs name=box_v1

这几个参数就是当时实际用的值（从 runs_box_v1.log 里读出来的）：
imgsz=480、epochs=100、batch=16、单卡 device=0、workers=0、seed=0。
其余如 mosaic=1.0、fliplr=0.5、patience=100、amp=True 都是 Ultralytics 默认值，没改。

产物落在 runs/box_v1/weights/。

## 第三步：上板

训练出的 .pt 要转成 K230 能跑的 .kmodel（nncase 工具链），放到板子 /sdcard/best.kmodel。
这一步当时是在另一台机器上做的，脚本没有存进本仓库，这里只记录产物是什么。
