
import os, sys, json, re, glob, shutil, pickle, logging, subprocess, time, ast
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch
_c = json.load(open("results/config.json"))
DATASETS, MODELS, N_TRAIN, N_TEST, SEED = _c["DATASETS"], _c["MODELS"], _c["N_TRAIN"], _c["N_TEST"], _c["SEED"]
EPOCHS = {tuple(k.split("|")): v for k, v in _c["EPOCHS"].items()}
HAS_GPU = torch.cuda.is_available()
cap = torch.cuda.get_device_capability(0)[0] if HAS_GPU else 0
DTYPE = "bfloat16" if cap >= 8 else "float32"          # authors' code defaults to bf16 on CUDA; T4 has no bf16
BATCH, GRAD_ACCUM = (32, 1) if cap >= 8 else (16, 2)   # effective batch size 32 in both cases
