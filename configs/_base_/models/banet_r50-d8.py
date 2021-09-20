# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='SFBHead',
        in_channels=256,
        in_index=4,
        channels=128,
        num_convs=1,
        kernel_size=3,
        concat_input=False,
        dropout_ratio=0,
        num_classes=19,
        use_boundary=False,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
            type='BFBHead',
            in_channels=16,
            in_index=5,
            channels=8,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0,
            num_classes=1,
            use_boundary=True,
            norm_cfg=norm_cfg,
            #sampler=dict(type='OHEMPixelSampler', thresh=0.9, min_kept=131072),
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.05)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
