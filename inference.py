
import argparse
from omegaconf import OmegaConf

from trainer import EvaluatorBase


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
            "--save_dir",
            type=str,
            default="train_result/test",  
            help="Folder to save the checkpoints and training log",
            )
    parser.add_argument(
            "--cfg_path",
            type=str,
            default="./configs/test.yaml",
            help="Configs of yaml file",
            )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_parser()
   
    configs = OmegaConf.load(args.cfg_path)

    # merge args to config
    for key in vars(args):
        if key in ['cfg_path', 'save_dir','iterative_refinement' ]:
            configs[key] = getattr(args, key)

    Evaluator = EvaluatorBase(configs)
    Evaluator.eval()

