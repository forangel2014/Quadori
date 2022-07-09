import imp
import torch
import numpy as np
from dpp_sampler import DPPsampler

if __name__ == "__main__":
    model_dir = "./model/dpp_encoder/"
    sampler = DPPsampler(0)
    L = sampler.get_L([["participate in", 1], ["will participate in", 1], ["has participated in", 1], ["how are you", 1]])
    loss = -torch.log(torch.det(L)/torch.det(L+torch.eye(L.shape[0]).cuda(0)))
    print(loss)