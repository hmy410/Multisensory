# encoding: utf-8
"""
用来训练source sep 模型

- 在sep_params.py更改模型训练时的参数，比如学习率、数据集路径、batch大小等等
-- 参数的设置基本都在 sep_params的 base函数里面
-- init_path 是设置基于哪个shift模型来训练
--

也可以用这个命令来训练：
num_gpus是使用的gpu数量，[N]这个是使用哪个gpu
python -c "import sep_params, sourcesep; sourcesep.train(sep_params.full(num_gpus=1), [1], restore = False)"

参考：
https://github.com/andrewowens/multisensory/issues/14
https://github.com/andrewowens/multisensory/issues/9
https://github.com/andrewowens/multisensory/issues/11#issuecomment-450376317
"""

import sourcesep, sep_params,new_sourcesep

clip_dur = 2.135
fn = getattr(sep_params, 'full')
pr = fn(vid_dur=clip_dur)
pr.batch_size=5
pr.init_path='../results/nets/sep/full/net.tf-160000'
pr.train_dir='../results/nets/sep/full_wave/training'
pr.resdir='../results/nets/sep/full_wave'
pr.summary_dir='../results/nets/sep/full_wave/summary'
pr.init_type='scratch'

# sourcesep.train(pr, 0, True, False, False)
new_sourcesep.train(pr, 0, False, False, False)
# def train(pr, gpus, restore = False, restore_opt = True, profile = False):
