_base_ = [
    '../../_base_/models/manet_r50.py', 
    # '../../_base_/schedules/sgd_manet_80e.py',
    '../../_base_/default_runtime.py'
]

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

log_level = 'INFO'
load_from = None
resume = False

epoch_max=80

# model = dict(
#     cls_head=dict(
#         label_smooth_eps=0.1,  # add label smooth
#         average_clips='prob'
#         ),
#     # add Mixup
#     # data_preprocessor=dict(
#     #     type='ActionDataPreprocessor',
#     #     mean=[123.675, 116.28, 103.53],
#     #     std=[58.395, 57.12, 57.375],
#     #     format_shape='NCHW',
#     #     blending=dict(
#     #         type='RandomBatchAugment',
#     #         augments=[
#     #             dict(type='MixupBlending', alpha=0.8, num_classes=52),
#     #             dict(type='CutmixBlending', alpha=1, num_classes=52)
#     #         ]),
#     #     ),
# )


# dataset settings
dataset_type = 'VideoDataset'
data_root = '/home/wangchen/projects/datasets/Microaction-52/train/'
data_root_val = '/home/wangchen/projects/datasets/Microaction-52/val/'
data_root_test = '/home/wangchen/projects/datasets/Microaction-52/val/'
ann_file_train = '/home/wangchen/projects/datasets/Microaction-52/annotations/train_list_videos.txt'
ann_file_val = '/home/wangchen/projects/datasets/Microaction-52/annotations/val_list_videos.txt'
ann_file_test = '/home/wangchen/projects/datasets/Microaction-52/annotations/val_list_videos.txt'

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', 
         clip_len=1,
         frame_interval=1,
         num_clips=8),
    dict(type='DecordDecode'),
    dict(type='Resize', 
         scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    # dict(type='Collect', keys=['imgs', 'label','emb'], meta_keys=[]),
    # dict(type='ToTensor', keys=['imgs', 'label','emb'])
    dict(type='PackActionInputs')
]
# test_pipeline = [
#     dict(type='DecordInit'),
#     dict(
#         type='SampleFrames',
#         clip_len=1,
#         frame_interval=1,
#         num_clips=8,
#         test_mode=True),
#     dict(type='DecordDecode'),
#     dict(type='Resize', scale=(-1, 256)),
#     dict(type='CenterCrop', crop_size=224),
#     # dict(type='Normalize', **img_norm_cfg),
#     dict(type='FormatShape', input_format='NCHW'),
#     # dict(type='Collect', keys=['imgs', 'label','emb'], meta_keys=[]),
#     # dict(type='ToTensor', keys=['imgs', 'label','emb'])
#     dict(type='PackActionInputs')
# ]

train_dataloader = dict(
    batch_size=20,
    num_workers=20,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=20,
    num_workers=20,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='AccMetric',
    metric_list=('top_k_accuracy', 'mean_class_accuracy'))
test_evaluator = val_evaluator
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=epoch_max, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=epoch_max,
        by_epoch=True,
        milestones=[30, 60],
        gamma=0.1)
]
optim_wrapper = dict(
    constructor='TSMOptimWrapperConstructor',  #TSMOptimizerConstructor
    paramwise_cfg=dict(fc_lr5=True),
    optimizer=dict(
        type='SGD', 
        lr=0.01/8, 
        momentum=0.9,
        weight_decay=0.0001),
    #  norm_decay_mult=0.0, bias_decay_mult=0.0
    clip_grad=dict(max_norm=20, norm_type=2))

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = '/home/wangchen/projects/mmaction2/work_dirs/manet/'