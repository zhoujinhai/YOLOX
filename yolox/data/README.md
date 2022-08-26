### 训练自己数据
#### 1. 准备数据集
采用labelme标注
利用labelme工程中自带的转换脚本将其转换成coco数据样式

#### 2. 数据存放格式
dataDir
   - annotations
       - annotations_train.json
       - annotations_val.json
   - JPEGImages
       - *.jpg
       
备注：按coco标准来讲，JPEGImages目录下还应分成train和val两个目录，这里修改了datasets/coco.py文件，所以没有再区分
```python
# img_file = os.path.join(self.data_dir, self.name, file_name)
img_file = os.path.join(self.data_dir, file_name)
# print("********", os.path.isfile(img_file))
img_file = img_file.replace("\\", "/")
```
       
#### 3. 新建自己参数脚本
在根目录下的exps目录下新建Tooth.py，依据实际情况设置相应参数
```python
#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33    # s: 0.33, m: 0.67, l: 1.0, x: 1.33
        self.width = 0.50    # s: 0.50, m: 0.75, l: 1.0, x: 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "/home/heygears/jinhai_zhou/data/2D_detect/tooth"
        self.train_ann = "annotations_train.json"
        self.val_ann = "annotations_val.json"

        self.num_classes = 2   # include background

        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 1

        self.cls_names = (
            "background",
            "tooth",
        )

```

#### 4.训练
进入根目录下的tools目录
执行：
```bash
python train.py -f ../exps/Tooth.py -d 0 -b 64 --fp16 -o -c /path/to/yolox_s.pth
```
其中-d后面数字表示GPU编号， -b 后面数字表示批量数据大小， --fp16表示混合精度训练  后面的pth文件为预训练权重

#### 5.测试
修改tools目录下demo.py，添加demo==images参数，支持图片批量测试

执行以下命令进行测试
```bash
python demo.py images -f ../exps/Tooth.py -n Tooth -c ../weights/Tooth_s.pth --path ../assets/tooth --conf 0.55 --nms 0.45 --tsize 640 --save_result --device cpu
```

#### 6.其他
训练过程中出现以下错误
<font color="red">RuntimeError: Ninja is required to load C++ extensions</font>

解决办法：
```bash
wget https://github.com/ninja-build/ninja/releases/download/v1.10.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 
```