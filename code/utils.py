# -*- coding: utf-8 -*-
import random
import numpy as np
import torch


try:
    from torch.profiler import profile, ProfilerActivity
    HAS_PROFILER = True
except Exception:
    profile, ProfilerActivity = None, None
    HAS_PROFILER = False


def enable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")
