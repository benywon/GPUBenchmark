# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/21 下午4:18
 @FileName: train.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

import argparse
import sys
import torch
sys.path.append("..")
import time
import torch.distributed as dist
from apex import amp
amp_handle = amp.init()

torch.distributed.init_process_group(backend='nccl',
                                     init_method='env://')

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
print(args.local_rank, dist.get_rank(), dist.get_world_size())
torch.cuda.set_device(args.local_rank)

