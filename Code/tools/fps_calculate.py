# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import time
import copy
import warnings
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model


def replace_ImageToTensor(pipelines):
    """Replace the ImageToTensor transform in a data pipeline to
    DefaultFormatBundle, which is normally useful in batch inference.

    Args:
        pipelines (list[dict]): Data pipeline configs.

    Returns:
        list: The new pipeline list with all ImageToTensor replaced by
            DefaultFormatBundle.
    """
    pipelines = copy.deepcopy(pipelines)
    for i, pipeline in enumerate(pipelines):
        if pipeline['type'] == 'MultiScaleFlipAug':
            assert 'transforms' in pipeline
            pipeline['transforms'] = replace_ImageToTensor(
                pipeline['transforms'])
        elif pipeline['type'] == 'ImageToTensor':
            warnings.warn(
                '"ImageToTensor" pipeline is replaced by '
                '"DefaultFormatBundle" for batch inference. It is '
                'recommended to manually replace it in the test '
                'data pipeline in your config file.', UserWarning)
            pipelines[i] = {'type': 'DefaultFormatBundle'}
    return pipelines

def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the results will be dumped '
              'into the directory as json'))
    parser.add_argument('--repeat-times', type=int, default=1)
    parser.add_argument('--model_basis', type=str, default='mmseg')
    args = parser.parse_args()
    return args


def main(args):

    cfg = Config.fromfile(args.config)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if args.work_dir is not None:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        json_file = osp.join(args.work_dir, f'fps_{timestamp}.json')
    else:
        # use config filename as default work_dir if cfg.work_dir is None
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        json_file = osp.join(work_dir, f'fps_{timestamp}.json')

    repeat_times = args.repeat_times
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False
    # cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    benchmark_dict = dict(config=args.config, unit='img / s')
    overall_fps_list = []
    for time_index in range(repeat_times):
        print(f'Run {time_index + 1}:')
        # cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=1,
            dist=False,
            shuffle=False)

        # build the model and load checkpoint
        cfg.model.train_cfg = None
        if args.model_basis == 'mmseg':
            model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        else:
            model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

        if 'checkpoint' in args and osp.exists(args.checkpoint):
            load_checkpoint(model, args.checkpoint, map_location='cpu')

        model = MMDataParallel(model, device_ids=[0])

        model.eval()

        # the first several iterations may be very slow so skip them
        num_warmup = 5
        pure_inf_time = 0
        total_iters = 200

        # benchmark with 200 image and take the average
        for i, data in enumerate(data_loader):

            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                model(return_loss=False, rescale=True, **data)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if i >= num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % args.log_interval == 0:
                    fps = (i + 1 - num_warmup) / pure_inf_time
                    print(f'Done image [{i + 1:<3}/ {total_iters}], '
                          f'fps: {fps:.2f} img / s')

            if (i + 1) == total_iters:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Overall fps: {fps:.2f} img / s\n')
                benchmark_dict[f'overall_fps_{time_index + 1}'] = fps
                benchmark_dict[f'overall_times_{time_index + 1}'] = 1000 / fps
                overall_fps_list.append(fps)
                break
    print(overall_fps_list)
    benchmark_dict['average_fps'] = round(np.mean(overall_fps_list), 2)
    benchmark_dict['fps_variance'] = round(np.var(overall_fps_list), 2)
    print(f'Average fps of {repeat_times} evaluations: '
          f'{benchmark_dict["average_fps"]}')
    print(f'The variance of {repeat_times} evaluations: '
          f'{benchmark_dict["fps_variance"]}')
    mmcv.dump(benchmark_dict, json_file, indent=4)


if __name__ == '__main__':
    args = parse_args()
    if args.model_basis == 'mmseg':
        from mmseg.datasets import build_dataloader, build_dataset
        from mmseg.models import build_segmentor
    else:
        from mmdet.datasets import build_dataloader, build_dataset
        from mmdet.models import build_detector
    main(args)
