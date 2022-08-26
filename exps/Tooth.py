#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33    # nano: 0.33, tiny:0.33,   s: 0.33, m: 0.67, l: 1.0, x: 1.33  # nano和tiny需调整self.input_size = (416, 416)
        self.width = 0.50    # nano: 0.25, tiny: 0.375, s: 0.50, m: 0.75, l: 1.0, x: 1.25
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
