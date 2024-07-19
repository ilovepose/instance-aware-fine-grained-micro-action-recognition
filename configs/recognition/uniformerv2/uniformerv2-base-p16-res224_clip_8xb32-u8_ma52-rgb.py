_base_ = ['../../_base_/default_runtime.py']
load_from = r'wegihts/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics400-rgb_20230313-75be0806.pth'
# model settings
num_frames = 32
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='UniFormerV2',
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        t_size=num_frames,
        dw_reduction=1.5,
        backbone_drop_path_rate=0.,
        temporal_downsample=False,
        no_lmhra=True,
        double_lmhra=True,
        return_list=[8, 9, 10, 11],
        n_layers=4,
        n_dim=768,
        n_head=12,
        mlp_factor=4.,
        drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5],
        clip_pretrained=True,
        pretrained='ViT-B/16'),
    cls_head=dict(
        type='UniFormerHead',
        dropout_ratio=0.5,
        num_classes=52,
        in_channels=768,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[114.75, 114.75, 114.75],
        std=[57.375, 57.375, 57.375],
        format_shape='NCTHW'))

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/home/wangchen/projects/datasets/Microaction-52/train/'  #_trace1_trace2
data_root_val = '/home/wangchen/projects/datasets/Microaction-52/val/'
data_root_test = '/home/wangchen/projects/datasets/Microaction-52/val/'
ann_file_train = '/home/wangchen/projects/datasets/Microaction-52/annotations/train_list_videos_aug.txt'  # train_trace1_trace2.txt
ann_file_val = '/home/wangchen/projects/datasets/Microaction-52/annotations/val_list_videos.txt'
ann_file_test = '/home/wangchen/projects/datasets/Microaction-52/annotations/val_list_videos.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    # dict(type='UniformSample', clip_len=num_frames, num_clips=1),
    # dict(type='DecordDecode'),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(
    #     type='PytorchVideoWrapper',
    #     op='RandAugment',
    #     magnitude=7,
    #     num_layers=4),
    # dict(type='RandomResizedCrop'),
    # dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecodeCrop', train=True, scale=(160, 320)),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    # dict(
    #     type='UniformSample', clip_len=num_frames, num_clips=1,
    #     test_mode=True),
    # dict(type='DecordDecode'),
    # dict(type='Resize', scale=(-1, 224)),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1, test_mode=True),
    dict(type='DecordDecodeCrop', train=True, scale=(160, 320)),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    # dict(
    #     type='UniformSample', clip_len=num_frames, num_clips=4,
    #     test_mode=True),
    # dict(type='DecordDecode'),
    # dict(type='Resize', scale=(-1, 224)),
    # dict(type='ThreeCrop', crop_size=224),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=4, test_mode=True),
    dict(type='DecordDecodeCrop', train=False, scale=(160, 320)),
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
    type='EpochBasedTrainLoop', max_epochs=55, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

base_lr = 1e-5
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=20, norm_type=2))

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
        T_max=50,
        eta_min_ratio=0.1,
        by_epoch=True,
        begin=5,
        end=55,
        convert_to_iter_based=True)
]

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=256)
