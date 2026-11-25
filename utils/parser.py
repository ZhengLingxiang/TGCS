import argparse
import os
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="Train or evaluate a TGCS classifier.")
    parser.add_argument('--config', default='cfgs/gpt/tgcs_scan_hardest.yaml', help='YAML configuration')
    parser.add_argument('--ckpts', help='Backbone checkpoint for training or trained classifier for evaluation')
    parser.add_argument('--train', action='store_true', help='Train TGCS (default)')
    parser.add_argument('--test', action='store_true', help='Evaluate a trained classifier')
    parser.add_argument('--finetune_model', action='store_true', help='Train the full model')
    parser.add_argument('--resume', action='store_true', help='Resume the selected training run')
    parser.add_argument('--tsne', action='store_true', help='Visualize classification features with t-SNE')
    parser.add_argument('--vote', action='store_true', help='Also evaluate with voting')
    parser.add_argument('--exp_name', default='experiment', help='Run name under experiments/')
    parser.add_argument('--launcher', choices=('none', 'pytorch'), default='none')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic cuDNN settings')
    parser.add_argument('--sync_bn', action='store_true', help='Use synchronized batch normalization in DDP')
    parser.add_argument('--val_freq', type=int, default=1, help='Validation interval in epochs')
    parser.add_argument('--way', type=int, default=-1)
    parser.add_argument('--shot', type=int, default=-1)
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--geometry-backend', choices=('cpu_cached', 'gpu_analytic', 'cpu_legacy'),
                        default='cpu_cached')
    parser.add_argument('--start_ckpts', help='Initial checkpoint for the pretraining runner')
    args = parser.parse_args()
    if args.test and (args.train or args.resume or args.finetune_model or args.tsne):
        parser.error('--test cannot be combined with --train, --resume, --finetune_model or --tsne')
    if args.tsne and (args.train or args.resume or args.finetune_model):
        parser.error('--tsne cannot be combined with training')
    if (args.test or args.tsne) and not args.ckpts:
        parser.error('Specify --ckpts for evaluation or visualization')
    if args.resume and args.start_ckpts:
        parser.error('--resume and --start_ckpts cannot be combined')
    if args.val_freq < 1:
        parser.error('--val_freq must be positive')

    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    if args.test:
        args.exp_name = 'test_' + args.exp_name
    config = Path(args.config)
    run_dir = Path('experiments') / config.stem / config.parent.stem
    args.experiment_path = str(run_dir / args.exp_name)
    args.tfboard_path = str(run_dir / 'TFBoard' / args.exp_name)
    args.log_name = config.stem
    Path(args.experiment_path).mkdir(parents=True, exist_ok=True)
    Path(args.tfboard_path).mkdir(parents=True, exist_ok=True)
    return args
