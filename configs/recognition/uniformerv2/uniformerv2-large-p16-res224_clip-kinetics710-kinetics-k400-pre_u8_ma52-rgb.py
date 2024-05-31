_base_ = ['../../_base_/default_runtime.py']

# model settings
num_frames = 8
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='UniFormerV2',
        input_resolution=224,
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        t_size=num_frames,
        dw_reduction=1.5,
        backbone_drop_path_rate=0.,
        temporal_downsample=False,
        no_lmhra=True,
        double_lmhra=True,
        return_list=[20, 21, 22, 23],
        n_layers=4,
        n_dim=1024,
        n_head=16,
        mlp_factor=4.,
        drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5],
        clip_pretrained=None),
    cls_head=dict(
        type='TimeSformerHead',
        dropout_ratio=0.5,
        num_classes=52,
        in_channels=1024,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[114.75, 114.75, 114.75],
        std=[57.375, 57.375, 57.375],
        format_shape='NCTHW'))

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/home/wangchen/projects/datasets/Microaction-52/train'
data_root_val = '/home/wangchen/projects/datasets/Microaction-52/val'
ann_file_train = '/home/wangchen/projects/datasets/Microaction-52/annotations/train_list_videos.txt'
ann_file_val = '/home/wangchen/projects/datasets/Microaction-52/annotations/val_list_videos.txt'
ann_file_test = '/home/wangchen/projects/datasets/Microaction-52/annotations/val_list_videos.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
# val_pipeline = [
#     dict(type='DecordInit', **file_client_args),
#     dict(
#         type='SampleFrames',
#         clip_len=8,
#         frame_interval=8,
#         num_clips=1,
#         test_mode=True),
#     dict(type='DecordDecode'),
#     dict(type='Resize', scale=(-1, 224)),
#     dict(type='CenterCrop', crop_size=224),
#     dict(type='FormatShape', input_format='NCTHW'),
#     dict(type='PackActionInputs')
# ]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='UniformSample', clip_len=num_frames, num_clips=4,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline)
)
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         ann_file=ann_file_val,
#         data_prefix=dict(video=data_root_val),
#         pipeline=val_pipeline,
#         test_mode=True))
test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True,
        delimiter=' '))
val_dataloader = test_dataloader

train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=100, 
    val_begin=1, 
    val_interval=1)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))
val_evaluator  = dict(type='AccMetric')
test_evaluator = dict(type='AccMetric')
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
