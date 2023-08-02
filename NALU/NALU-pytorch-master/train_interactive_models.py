import random
import numpy as np

import torch
import torch.nn.functional as F

from models import MLP, NAC, NALU
from genereate_data import generate_data

import pickle

NORMALIZE = True
NUM_LAYERS = 2
HIDDEN_DIM = 2
LEARNING_RATE = 1e-2
NUM_ITERS = int(1e2)
RANGE = [5, 10]
ARITHMETIC_FUNCTIONS = {
    'add': lambda x, y: x + y,
    'sub': lambda x, y: x - y,
    'mul': lambda x, y: x * y,
    'div': lambda x, y: x / y,
    'squared': lambda x, y: torch.pow(x, 2),
    'root': lambda x, y: torch.sqrt(x),
}


def train(model, optimizer, data, target, num_iters):
    for i in range(num_iters):
        out = model(data)
        loss = F.mse_loss(out, target)
        mea = torch.mean(torch.abs(target - out))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print(f"\t{i+1}/{num_iters}: loss: {loss.item():.7f} - mea: {mea.item():.7f}")


def main():
    save_dir = './trained_models/'

    models = [
        MLP(
            num_layers=NUM_LAYERS,
            in_dim=2,
            hidden_dim=HIDDEN_DIM,
            out_dim=1,
            activation='relu6',
        ),
        MLP(
            num_layers=NUM_LAYERS,
            in_dim=2,
            hidden_dim=HIDDEN_DIM,
            out_dim=1,
            activation='none',
        ),
        NAC(
            num_layers=NUM_LAYERS,
            in_dim=2,
            hidden_dim=HIDDEN_DIM,
            out_dim=1,
        ),
        NALU(
            num_layers=NUM_LAYERS,
            in_dim=2,
            hidden_dim=HIDDEN_DIM,
            out_dim=1
        ),
    ]

    for fn_str, fn in ARITHMETIC_FUNCTIONS.items():
        print(f'[*] Function: {fn_str}')

        # dataset
        X_train, y_train= generate_data(
            num_data=500,
            dim=100, num_sum=5, fn=fn,
            support=RANGE,
        )

        # others
        for net in models:
            model_name = net.__str__().split("(")[0]
            print(f'\tTraining {model_name}...')
            optim = torch.optim.RMSprop(net.parameters(), lr=LEARNING_RATE)
            train(net, optim, X_train, y_train, NUM_ITERS)

            if model_name == "MLP" and net.activation:
                model_name += f"_{net.activation.__str__().split('(')[0]}"
            name = f"{fn_str}_{model_name}_trained"
            f = open(f"{save_dir}{name}.obj", "wb")
            pickle.dump(net, f)
            f.close()


if __name__ == '__main__':
    main()
