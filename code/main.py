# -*- coding: utf-8 -*-
import argparse

from train import train
from utils import enable_tf32


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/MovieLens-100K")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--mu", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--steps_per_epoch", type=int, default=300)

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Stop after this many evaluations without improvement. Set <=0 to disable.")
    parser.add_argument("--early_stop_delta", type=float, default=1e-6, help="Minimum N@20 improvement required to reset early stopping.")

    parser.add_argument("--subgraph_users", type=int, default=3000)
    parser.add_argument("--subgraph_items", type=int, default=5000)
    parser.add_argument("--subgraph_item_pad", type=int, default=600)

    parser.add_argument("--entmax_alpha", type=float, default=1.8)
    parser.add_argument("--cl_weight", type=float, default=0.05)
    parser.add_argument("--cl_tau", type=float, default=0.2, help="InfoNCE temperature")
    parser.add_argument("--lightgcn_layers", type=int, default=4, help="Layers for auxiliary LightGCN view")

    parser.add_argument("--param_reg", type=float, default=1e-6)
    parser.add_argument("--init", default="xavier")
    parser.add_argument("--init_std", type=float, default=0.1)

    parser.add_argument("--dns_k", type=int, default=8, help="Number of candidates for DNS. Set <=1 to use random sampling.")

    parser.add_argument("--use_neumann", action="store_true")
    parser.add_argument("--neumann_K", type=int, default=3)
    parser.add_argument("--sparse_rtilde", action="store_true")
    parser.add_argument("--eval_use_neumann", action="store_true")
    parser.add_argument("--eval_use_explicit", action="store_false", dest="eval_use_neumann")
    parser.set_defaults(eval_use_neumann=True)
    parser.add_argument("--report_flops", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    enable_tf32()
    train(parse_args())
