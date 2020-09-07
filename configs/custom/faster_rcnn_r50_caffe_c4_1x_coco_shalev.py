_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_c4.py',
    # '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Duplicate'),
    # dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Duplicate'),
            # dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
classes = ['bear']
dataset_type = 'CocoDataset'
data_root = '/home/shalev/downloads/1pic_coco/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file='/home/shalev/downloads/annotations/instances_val2017_1pic.json',
        # ann_file = '/home/shalev/downloads/annotations/instances_val2017_new.json',
        classes=classes,
        img_prefix=data_root, #+ 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='/home/shalev/downloads/annotations/instances_val2017_1pic.json',
        # ann_file = '/home/shalev/downloads/annotations/instances_val2017_new.json',
        classes=classes,
        img_prefix=data_root, #+ 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/home/shalev/downloads/annotations/instances_val2017_1pic.json',
        # ann_file = '/home/shalev/downloads/annotations/instances_val2017_new.json',
        classes=classes,
        img_prefix=data_root, #+ 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=50, metric='bbox')
checkpoint_config = dict(interval=50)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
total_epochs = 5000
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
work_dir = '/media/shalev/98a3e66d-f664-402a-9639-15ec6b8a7150/work_dirs/try2'
resume_from='/media/shalev/98a3e66d-f664-402a-9639-15ec6b8a7150/work_dirs/try2/latest.pth'