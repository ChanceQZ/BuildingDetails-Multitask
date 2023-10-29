# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.0001,
    # step=[27, 33]
    # step=[65, 71]
    step=[90, 106]
    )
runner = dict(type='EpochBasedRunner', max_epochs=108)
