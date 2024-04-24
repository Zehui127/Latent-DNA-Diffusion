import argparse, os, sys
import time, datetime
import yaml, easydict
import torch
# Datasets
from src.datasets.reference_loader import *
# Sessions
import src.sessions.vanilla_vae as vanilla_vae
import src.sessions.unconditional_unet as un_unet
# Utils
from config import *
from src.utils.metrics import *

def build_parser(root):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vanilla_vae', help='model to train')
    parser.add_argument('--dataset', type=str, default='reference', help='dataset to train on')
    parser.add_argument('--cfgdir', type=str, default=root+'/src/configs', help='path to config file')
    parser.add_argument('--cfgname', type=str, default='', help='name of config file')
    parser.add_argument('--cptname', type=str, default='', help='name of checkpoint file')
    parser.add_argument('--cptnumber', type=int, default=1, help='epoch number of first iteration')
    parser.add_argument('--tgtdir', type=str, default=root+'/saved_models', help='path to save model')
    parser.add_argument('--gen_vae_path', type=str, default=root+'/saved_models', help='path to load vae model')
    parser.add_argument('--gen_unet_path', type=str, default=root+'/saved_models', help='path to load unet model')
    return parser

def prepare_config(config_path, save_path):
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = easydict.EasyDict(cfg)
    print(cfg)
    config = TrainerConfig(save_path, cfg['trainer_cfg'], cfg['loader_cfg'], cfg['model_cfg'])

    return config

if __name__ == "__main__":
    parser = build_parser(os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    path = os.path.join(args.cfgdir, args.model + '.yaml') if not args.cfgname else os.path.join(args.cfgdir, args.cfgname + '.yaml')
    config = prepare_config(path, args.tgtdir)
    # create the target directory if it does not exist
    if not os.path.exists(args.tgtdir):
        os.makedirs(args.tgtdir)

    if args.model == 'vanilla_vae':
        vanilla_vae.train(config, args.cptname, args.cptnumber)
    if args.model == 'un_unet':
        un_unet.train_loop(config)
    if args.model == 'generate':
        # args.cptname: the checkpoint path of UNet
        un_unet.evaluate(config,args.gen_vae_path,args.gen_unet_path)
