
import wandb
wandb.init(project="my-test-project")
from mmpose.datasets import build_dataset
from mmpose.models import build_posenet, losses
from mmpose.apis import train_model
import mmcv

#import Config_constructing
from mmpose.datasets.datasets.top_down import TopDownCocoDataset

#from analysis import plot_curve, parse_args, load_json_logs


#Config
from mmcv import Config

#cfg = Config.fromfile(
#    './configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'
#)
cfg = Config.fromfile(
    './configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest50_coco_384x288.py'
)

#wandb.config = {
#  "learning_rate": 0.001,
#  "epochs": 100,
#  "batch_size": 128
#}

# set basic configs
cfg.data_root = 'data/coco'
#cfg.work_dir = '/local/fshafi/mmpose/projectB/work_dirs/hrnet_w32_coco_256x192'
cfg.work_dir = '/local/fshafi/mmpose/mmpose/Training_w_new_loss/work_dirs/resnest50_coco_384x288'

cfg.gpu_ids = range(1)
cfg.seed = 0

# set log interval
cfg.log_config.interval = 100

# set evaluation configs
cfg.evaluation.interval = 1
#cfg.evaluation.metric = 'pck'
#cfg.evaluation.save_best = 'pck'

# set loss curve and validation result
cfg.log_config=dict(
    interval=100,
    hooks=[
        # The default hook to output information to screen and local files
        dict(type='TextLoggerHook'),
        # you can add the following hook to visualize with Wandb
        dict(type='WandbLoggerHook'),  
        # You can add the following hook to visualize with Tensorboard
        dict(type='TensorboardLoggerHook')
    ])

# set workflow
#cfg.workflow = [('train',1),('val',1)]

# set learning rate policy
cfg.lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.001,
    step=[17, 35])
cfg.total_epochs = 100

# set batch size
cfg.data.samples_per_gpu = 16
cfg.data.val_dataloader = dict(samples_per_gpu=16)
cfg.data.test_dataloader = dict(samples_per_gpu=16)

# set dataset configs
cfg.data.train.type = 'TopDownCocoDataset'
cfg.data.train.ann_file = f'{cfg.data_root}/annotations/person_keypoints_train2017.json'
cfg.data.train.img_prefix = f'{cfg.data_root}/train2017/'

cfg.data.val.type = 'TopDownCocoDataset'
cfg.data.val.ann_file = f'{cfg.data_root}/annotations/person_keypoints_val2017.json'
cfg.data.val.img_prefix = f'{cfg.data_root}/val2017/'

cfg.data.test.type = 'TopDownCocoDataset'
cfg.data.test.ann_file = f'{cfg.data_root}/annotations/person_keypoints_val2017.json'
cfg.data.test.img_prefix = f'{cfg.data_root}/val2017/'




# set model configs
#cfg.model.keypoint_head.loss_keypoint.type = 'MyLoss'
# model settings

cfg.channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ]) 

# model settings
""" cfg.model = dict(
    type='TopDown',
    pretrained='mmcls://resnest50',
    backbone=dict(type='ResNeSt', depth=50),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=2048,
        out_channels=cfg.channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='MyLoss', use_target_weight=False)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11)) """ 
cfg.loss_keypoint=dict(type='AdaptiveWingLoss', use_target_weight=False)
#####################################################################################################
import warnings

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         get_dist_info)
from mmcv.utils import digit_version

from mmpose.core import DistEvalHook, EvalHook, build_optimizers
from mmpose.core.distributed_wrapper import DistributedDataParallelWrapper
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.utils import get_root_logger

try:
    from mmcv.runner import Fp16OptimizerHook
except ImportError:
    warnings.warn(
        'Fp16OptimizerHook from mmpose will be deprecated from '
        'v0.15.0. Please install mmcv>=1.1.4', DeprecationWarning)
    from mmpose.core import Fp16OptimizerHook


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.
    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()

def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """Train model entry function.
    Args:
        model (nn.Module): The model to be trained.
        dataset (Dataset): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    
    logger = get_root_logger('result',cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(
            seed=cfg.get('seed'),
            drop_last=False,
            dist=distributed,
            num_gpus=len(cfg.gpu_ids)),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'samples_per_gpu',
                   'workers_per_gpu',
                   'shuffle',
                   'seed',
                   'drop_last',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }

    # step 2: cfg.data.train_dataloader has highest priority
    train_loader_cfg = dict(loader_cfg, **cfg.data.get('train_dataloader', {}))

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # determine whether use adversarial training precess or not
    use_adverserial_train = cfg.get('use_adversarial_train', False)

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel

        if use_adverserial_train:
            # Use DistributedDataParallelWrapper for adversarial training
            model = DistributedDataParallelWrapper(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        if digit_version(mmcv.__version__) >= digit_version(
                '1.4.4') or torch.cuda.is_available():
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        else:
            warnings.warn(
                'We recommend to use MMCV >= 1.4.4 for CPU training. '
                'See https://github.com/open-mmlab/mmpose/pull/1157 for '
                'details.')

    # build runner
    optimizer = build_optimizers(model, cfg.optimizer)

    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    if use_adverserial_train:
        # The optimizer step process is included in the train_step function
        # of the model, so the runner should NOT include optimizer hook.
        optimizer_config = None
    else:
        # fp16 setting
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
        elif distributed and 'type' not in cfg.optimizer_config:
            optimizer_config = OptimizerHook(**cfg.optimizer_config)
        else:
            optimizer_config = cfg.optimizer_config

    custom_hooks_cfg = cfg.get('custom_hooks', None)
    if custom_hooks_cfg is None:
        custom_hooks_cfg = cfg.get('custom_hooks_config', None)
        if custom_hooks_cfg is not None:
            warnings.warn(
                '"custom_hooks_config" is deprecated, please use '
                '"custom_hooks" instead.', DeprecationWarning)

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=custom_hooks_cfg)

    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        eval_cfg = cfg.get('evaluation', {})
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            drop_last=False,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)










#####################################################################################

# build dataset
datasets = [build_dataset(cfg.data.train)]

# build model
model = build_posenet(cfg.model)


# create work_dir
mmcv.mkdir_or_exist(cfg.work_dir)

# train model
train_model(
    model, datasets, cfg, distributed=False, validate=True, meta=dict())

#args = parse_args()
#log_dicts = load_json_logs("projectB/work_dirs/hrnet_w32_coco_256x192/None.log.json")
#plot_curve(log_dicts,args)

#wandb.log({"train-loss":losses, "epoch":1})
# Optional
wandb.watch(model)