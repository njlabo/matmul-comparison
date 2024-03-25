# Don't edit this file! This was automatically generated from "test_all.ipynb".

import torch
from torch import nn
import struct
import numpy as np
import subprocess
import time
import pandas as pd

def serialize_fp32(file, tensor):
    ''' Write one fp32 tensor to file that is open in wb mode '''
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def save_model_and_data(data, model):
    f = open("model.pt", "wb")
    for p in model.parameters():
        serialize_fp32(f, p)
    f.close()

    f = open("data.txt", "wb")
    serialize_fp32(f, data)
    f.close()

def execute_commands(commands):
    f = open("stdout.txt", "wb")
    durations = []
    for cmd in commands:
        start_time = time.time()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.PIPE)
        proc.wait()
        duration = time.time() - start_time
        durations.append(duration)
    f.close()

    return durations

def calculate_time(data, model, commands):
    save_model_and_data(data, model)
    start_time = time.time()
    ref = model(data).detach().numpy().flatten()
    print("Python: {:.3f} seconds".format(time.time() - start_time))

    # C: Four execution with and without OpenMP and BLAS support
    durations = execute_commands(commands)

    # check results match reference values
    res = np.loadtxt("stdout.txt").reshape(4, len(ref))
    for i in range(4):
        assert np.allclose(res[i], ref, rtol=1e-05, atol=1e-06)

    # display durations
    df = pd.DataFrame(np.array(durations).reshape(2,2)).round(3)
    df.columns = ["Without BLAS", "With BLAS"]
    df.index = ["Without OpenMP", "With OpenMP"]
    print(df, "\n")

def test_linear():
    dim = 4096
    data = torch.rand(dim)
    fc = nn.Linear(dim, dim, bias=True)
    model = nn.Sequential(*[fc for _ in range(100)])
    # C: Four execution with and without OpenMP and BLAS support
    commands = ["./run-linear", ["./run-linear", "blas"], "./run-linear-p", ["./run-linear-p", "blas"]]
    calculate_time(data, model, commands)

def test_conv():
    nch_in, h, w = 3, 28, 28
    nch_hid, ks, st, pad = 64, 3, 1, 1
    data = torch.rand(nch_in * h * w).view(1, nch_in, h, w)
    conv_in = nn.Conv2d(nch_in, nch_hid, ks, st, pad, bias=True)
    conv = nn.Conv2d(nch_hid, nch_hid, ks, st, pad, bias=True)
    model = nn.Sequential(conv_in, *[conv for _ in range(100)])

    # C: Four execution with and without OpenMP and BLAS support
    commands = ["./run-conv", ["./run-conv", "blas"], "./run-conv-p", ["./run-conv-p", "blas"]]
    calculate_time(data, model, commands)
