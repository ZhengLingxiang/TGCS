import os
from tools import test_run_net as test_net
from tools import finetune_run_net as finetune
from tools import module_run_net as module_tune
from tools.runner_finetune import test_tsne
from utils import parser, dist_utils, misc
from utils.logger import get_root_logger
from utils.config import get_config, log_args_to_file, log_config_to_file
import time
import torch
from tensorboardX import SummaryWriter


def main():
    args = parser.get_args()
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)

    config = get_config(args, logger = logger)
    config.model.geometry_backend = args.geometry_backend
    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs // world_size
        config.dataset.val.others.bs = config.total_bs // world_size
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs // world_size
    else:
        config.dataset.train.others.bs = config.total_bs
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs
        config.dataset.val.others.bs = config.total_bs * 2

        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs * 2

    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)

    logger.info(f'Distributed training: {args.distributed}')
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.cuda.current_device()

    if config.dataset.train._base_.NAME == 'ModelNetFewShot':
        if args.way < 1 or args.shot < 1 or args.fold < 0:
            raise ValueError('Specify --way, --shot and --fold for a few-shot configuration.')
    if args.shot != -1:
        config.dataset.train.others.shot = args.shot
        config.dataset.train.others.way = args.way
        config.dataset.train.others.fold = args.fold
        config.dataset.val.others.shot = args.shot
        config.dataset.val.others.way = args.way
        config.dataset.val.others.fold = args.fold

    if args.test:
        return test_net(args, config)
    if args.tsne:
        return test_tsne(args, config)
    if config.task != 'classification':
        raise ValueError(f'Unsupported training task: {config.task}')

    train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
    val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
    run = finetune if args.finetune_model else module_tune
    return run(args, config, train_writer, val_writer)


if __name__ == '__main__':
    main()
