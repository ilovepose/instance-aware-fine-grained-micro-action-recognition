_base_ = [
    '../../_base_/models/swin_tiny.py', '../../_base_/default_runtime.py'
]

# load_from = '/home/wangchen/projects/mmaction2/work_dirs/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-40e_ma52-rgb/best_acc_f1_mean_6741.pth'

model = dict(
    backbone=dict(
        pretrained=  # noqa: E251
        # 'https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_tiny_patch4_window7_224.pth'  # noqa: E501
        'weights/swin_tiny_patch4_window7_224.pth'
    ),
    cls_head=dict(
        type='MANet3DHead',
        in_channels=768,
        num_classes=52,
        # loss_cls=dict(type='CoarseFocalLoss'),
        label_smooth_eps=0.2),
    )

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/home/wangchen/projects/datasets/Microaction-52/train/'
data_root_val = '/home/wangchen/projects/datasets/Microaction-52/val/'
data_root_test = '/home/wangchen/projects/datasets/Microaction-52/val/'
ann_file_train = '/home/wangchen/projects/datasets/Microaction-52/annotations/train_list_videos.txt'
ann_file_val = '/home/wangchen/projects/datasets/Microaction-52/annotations/val_list_videos.txt'
ann_file_test = '/home/wangchen/projects/datasets/Microaction-52/annotations/val_list_videos.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecodeCrop', train=True, scale=(160, 320)),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1, test_mode=True),
    dict(type='DecordDecodeCrop', train=False, scale=(160, 320)),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=4, test_mode=True),
    dict(type='DecordDecodeCrop', train=False, scale=(160, 320)),
    # dict(type='DecordDecode'),
    # dict(type='Resize', scale=(-1, 224)),
    # dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_test),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric', metric_list=('f1_mean', 'top_k_accuracy', 'mean_class_accuracy'))
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=40, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.02),
    constructor='SwinOptimWrapperConstructor',
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.),
        relative_position_bias_table=dict(decay_mult=0.),
        norm=dict(decay_mult=0.),
        backbone=dict(lr_mult=0.1)))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=2.5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=40,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=40)
]

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=8)
