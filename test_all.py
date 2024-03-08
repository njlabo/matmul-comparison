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

def test_linear():
    inp = 4096
    out = 4096

    fc = nn.Linear(inp, out, bias=True)

    model = nn.Sequential(*[fc for _ in range(1000)])
    f = open("model.pt", "wb")
    for p in model.parameters():
        serialize_fp32(f, p)
    f.close()

    data = torch.rand(inp)
    f = open("data.txt", "wb")
    serialize_fp32(f, data)
    f.close()

    # measure the time it takes to perform linear operations 
    # Python
    start_time = time.time()
    ref = model(data).detach().numpy()
    print("Python: {:.3f} seconds".format(time.time() - start_time))

    # write outputs C to make sure later the results match
    f = open("stdout.txt", "wb")

    # C: Four execution with and without OpenMP and BLAS support
    commands = ["./run-linear", ["./run-linear", "blas"], "./run-linear-p", ["./run-linear-p", "blas"]]
    durations = []
    for command in commands:
        start_time = time.time()
        proc = subprocess.Popen(command, stdout=f, stderr=subprocess.PIPE)
        proc.wait()
        duration = time.time() - start_time
        durations.append(duration)
        print("{}: {:.3f} seconds".format(command, duration))

    f.close()

    # ensure the results match
    res = np.loadtxt("stdout.txt").reshape(4, out)
    for i in range(4):
        assert np.allclose(res[i], ref, rtol=1e-05, atol=1e-06) # Python and C non-optimized

    df = pd.DataFrame(np.array(durations).reshape(2,2)).round(3)
    df.columns = ["Without BLAS", "With BLAS"]
    df.index = ["Without OpenMP", "With OpenMP"]
    print()
    print(df)

def test_conv():
    conv_in = nn.Conv2d(3, 64, 3, 1, 1, bias=True)
    conv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)

    model = nn.Sequential(conv_in, *[conv for _ in range(100)])
    f = open("model.pt", "wb")
    for p in model.parameters():
        serialize_fp32(f, p)
    f.close()

    data = torch.rand(3 * 28 * 28).view(1, 3, 28, 28)
    f = open("data.txt", "wb")
    serialize_fp32(f, data)
    f.close()

    # measure the time it takes to perform convolutional operations 
    # Python
    start_time = time.time()
    ref = model(data).detach().numpy()
    print("Python: {:.3f} seconds".format(time.time() - start_time))

    # write outputs C to make sure later the results match
    f = open("stdout.txt", "wb")

    # C: Four execution with and without OpenMP and BLAS support
    commands = ["./run-conv", ["./run-conv", "blas"], "./run-conv-p", ["./run-conv-p", "blas"]]
    durations = []
    for command in commands:
        start_time = time.time()
        proc = subprocess.Popen(command, stdout=f, stderr=subprocess.PIPE)
        proc.wait()
        duration = time.time() - start_time
        durations.append(duration)
        print("{}: {:.3f} seconds".format(command, duration))

    f.close()

    # ensure the results match
    res = np.loadtxt("stdout.txt").reshape(4, 64, 28, 28)
    for i in range(4):
        assert np.allclose(res[i], ref, rtol=1e-05, atol=1e-06) # Python and C non-optimized

    df = pd.DataFrame(np.array(durations).reshape(2,2)).round(3)
    df.columns = ["Without BLAS", "With BLAS"]
    df.index = ["Without OpenMP", "With OpenMP"]
    print()
    print(df)
