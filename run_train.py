import argparse
import importlib
import utils
import subprocess
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="semi-sup")
    parser.add_argument("-c", "--config", type=str, required=True,help='path to config file')
    parser.add_argument("-exp_name", type=str, help='name of this run')
    parser.add_argument("--log_dir", type=str, default='logs')
    parser.add_argument("--weights_dir", type=str, default='weights')
    parser.add_argument("--plots_dir", type=str, default='plots')
    parser.add_argument('--tb_port', action="store", type=int, default=6006, help="tensorboard port")

    args = parser.parse_args()
    if not args.exp_name:
        args.exp_name = args.config

    config = importlib.import_module('configs.{}'.format(args.config)).config
    utils.initialize_dirs_and_files(args, config)
    
    # kill existing tensorboard processes on port (in order to refresh)
    utils.kill_processes_on_port(args.tb_port)

    # start tensorboard
    env = dict(os.environ)   # Make a copy of the current environment
    subprocess.Popen('tensorboard --port {} --logdir ./{}'.format(args.tb_port, args.log_dir), env=env, shell=True) 

    utils.set_seed(config.seed)
    trainer = config.trainer(config)
    trainer.run()
