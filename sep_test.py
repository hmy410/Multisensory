# encoding: utf-8
"""
testing sep model
输出准确率
"""
# import shift_net, shift_params, numpy as np
import numpy as np
import sep_params
import sourcesep
# import my_shift_net,my_shift_net_v
import time
import os
import random
import tensorflow as tf


#model_file = '../results/nets/shift/net.tf-650000'
# model_file = '../data/scratch/shift_test/shift-lowfps_original/training/net.tf-30000'
# model_file = '../data/scratch/shift_test/shift-lowfps_transformer/training/net.tf-30000'
# model_file ='../data/scratch/shift/shift-lowfps_transformer-v_sgd_lr-1e-2_grad-clip5_no-augment_batch15_smaller-train/training/net.tf-180000'

import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("--resdir", default='/home/hemy/multisensory/results/nets/sep/full/training', type=str,
                    help="The save model dir. ")
parser.add_argument("--model_type", default=None, type=str,
                    help="Model type |shift_net| my_shift_net| my_shift_net_v")
parser.add_argument("--train_type", default=None, type=str,
                    help="training type |shift_lowfps |shift_v1")
parser.add_argument("--init_path", default=None, type=str,
                    help="Which model to use for initialing model.")
parser.add_argument("--model_file", default=None, type=str,
                    help="Which model to use for testing model.")
parser.add_argument("--train_list", default='../data/sep_voxcleb/train/tf', type=str,
                    help="The training dataset file, which contrains many .tf files path.")
parser.add_argument("--test_list", default='../data/sep_voxcleb/test/tf', type=str,
                    help="The testing dataset file, which contrains many .tf files path. It is not used when training")
parser.add_argument("--gpu", default="0,1,2,3", type=str,
                    help="which gpus to use")

parser.add_argument("--do_shift", default=False, type=str,
                    help="Whether to shift. ")
parser.add_argument("--restore", default=False, type=bool,
                    help="Whether to restore from last checkpoint. ")
parser.add_argument('--train_iters', type=int, default=30000,
                    help="Total Training iteration")
parser.add_argument("--opt_method", default='momentum', type=str,
                    help="Optimizer |adam|momentum")
parser.add_argument("--base_lr", default=5e-5, type=float,
                    help="The initial learning rate.")
parser.add_argument("--grad_clip", default=5.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--weight_decay", default=1e-5, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument('--check_iters', type=int, default=500,
                    help="Save checkpoint every X updates steps.")
parser.add_argument("--augment_ims", default=False, type=bool,
                    help="Whether do image augment, such as crop, rotate...")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")

parser.add_argument('--trf_hidden_units', type=int, default=512,
                    help="transformer alias = C  # 隐藏层维度")
parser.add_argument('--trf_num_blocks', type=int, default=1,
                    help="transformer number of encoder/decoder blocks   block个数.")
parser.add_argument('--trf_num_heads', type=int, default=4,
                    help="transformer head 个数")
parser.add_argument('--trf_dropout_rate', type=float, default=0.1,
                    help="transformer dropout rate")

args = parser.parse_args()

# pr = shift_params.shift_v1()


# if args.train_type == 'shift_lowfps':
#     fn = getattr(shift_params, 'shift_lowfps') # shift_lowfps shift_v1
# elif args.train_type == 'shift_v1':
#     fn = getattr(shift_params, 'shift_v1') # shift_lowfps shift_v1

fn = getattr(sep_params, 'full')

pr = fn()
pr.resdir = args.resdir
pr.opt_method = args.opt_method
pr.init_path = '../results/nets/shift/net.tf-650000'
pr.train_list = args.train_list
pr.test_list = args.test_list
pr.base_lr = args.base_lr
pr.grad_clip = args.grad_clip
pr.weight_decay = args.weight_decay
pr.check_iters = args.check_iters
pr.model_file='/home/hemy/multisensory/results/nets/sep/full/training/net-tf.58000'
# pr.trf_hidden_units = args.trf_hidden_units
# pr.trf_num_blocks = args.trf_num_blocks
# pr.trf_num_heads = args.trf_num_heads
# pr.trf_dropout_rate = args.trf_dropout_rate
# pr.do_shift = args.do_shift

if args.do_shift=='True':
    pr.do_shift=True
elif args.do_shift=='False':
    pr.do_shift=False

print('pr.test_list:',pr.test_list)
print('model_file:',pr.model_file)
print('pr.do_shift:',pr.do_shift)

random.seed(args.seed)
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

# if args.model_type == 'my_shift_net_v':
#     clf = my_shift_net_v.NetClf(pr, args.model_file, gpu=args.gpu)
# elif args.model_type == 'shift_net':
#     clf = shift_net.NetClf(pr, args.model_file, gpu=args.gpu)
# elif args.model_type == 'my_shift_net':
#     clf = my_shift_net.NetClf(pr, args.model_file, gpu=args.gpu)

clf = sourcesep.NetClf(pr, pr.resdir, gpu = args.gpu)

start_time = time.time()

accuray = clf.test_accuracy()

end_time = time.time()


print('pr.test_list:',pr.test_list)
print('model_file:',args.model_file)
print('pr.do_shift:',pr.do_shift)

print('accuray:',accuray)
print('cost time: {} s'.format(end_time-start_time))