_base_ = ['../../_base_/default_runtime.py']
load_from = r"weights/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-3e7f93b2.pth"
# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='VisionTransformer',
        img_size=224,
        patch_size=16,
        embed_dims=768,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=16,
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(
        type='TimeSformerHead',
        num_classes=52,
        in_channels=768,
        average_clips='prob',
        loss_cls=dict(type='CoarseFocalLoss')),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/home/wangchen/projects/datasets/Microaction-52/train_trace1_trace2/'  #
data_root_val = '/home/wangchen/projects/datasets/Microaction-52/val/'
data_root_test = '/home/wangchen/projects/datasets/Microaction-52/test/'
ann_file_train = '/home/wangchen/projects/datasets/Microaction-52/annotations/train_list_videos_aug3.txt'  #train_trace1_trace2.txt
ann_file_val = '/home/wangchen/projects/datasets/Microaction-52/annotations/val_list_videos.txt'
ann_file_test = '/home/wangchen/projects/datasets/Microaction-52/annotations/test_list_videos.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=1),
    dict(type='DecordDecodeCrop', train=True, scale=(160, 320)),  #192, 384
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=1, test_mode=True),
    dict(type='DecordDecodeCrop', train=False, scale=(160, 320)),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=4, test_mode=True),
    dict(type='DecordDecodeCrop', train=False, scale=(160, 320)),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16,
    num_workers=16,
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
    type='EpochBasedTrainLoop', max_epochs=20, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optim_wrapper = dict(
#     type='AmpOptimWrapper',
#     optimizer=dict(
#         type='AdamW', lr=0.0005, betas=(0.9, 0.999), weight_decay=0.05),
#     constructor='SwinOptimWrapperConstructor',
#     paramwise_cfg=dict(
#         absolute_pos_embed=dict(decay_mult=0.),
#         relative_position_bias_table=dict(decay_mult=0.),
#         norm=dict(decay_mult=0.),
#         backbone=dict(lr_mult=0.1),
#         ))

# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=0.1,
#         by_epoch=True,
#         begin=0,
#         end=2.5,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingLR',
#         T_max=40,
#         eta_min=0,
#         by_epoch=True,
#         begin=0,
#         end=40)
# ]
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=2.5e-4, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg=dict(
        decay_rate=0.9,
        decay_type='layer_wise',
        num_layers=12,
        backbone=dict(lr_mult=0.1),
    ),
    clip_grad=dict(max_norm=40, norm_type=2))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=15,
        eta_min=0,
        by_epoch=True,
        begin=5,
        end=20,
        convert_to_iter_based=True)
]

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=64)
