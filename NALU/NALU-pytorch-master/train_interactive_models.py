import random
import numpy as np

import torch
import torch.nn.functional as F

from models import MLP, NAC, NALU

import pickle

NORMALIZE = True
NUM_LAYERS = 2
HIDDEN_DIM = 2
LEARNING_RATE = 1e-2
NUM_ITERS = int(1e5)
RANGE = [5, 10]
ARITHMETIC_FUNCTIONS = {
    'add': lambda x, y: x + y,
    'sub': lambda x, y: x - y,
    'mul': lambda x, y: x * y,
    'div': lambda x, y: x / y,
    'squared': lambda x, y: torch.pow(x, 2),
    'root': lambda x, y: torch.sqrt(x),
}


def generate_data(num_train, num_test, dim, num_sum, fn, support):
    data = torch.FloatTensor(dim).uniform_(*support).unsqueeze_(1)
    X, y = [], []
    for i in range(num_train + num_test):
        idx_a = random.sample(range(dim), num_sum)
        idx_b = random.sample([x for x in range(dim) if x not in idx_a], num_sum)
        a, b = data[idx_a].sum(), data[idx_b].sum()
        X.append([a, b])
        y.append(fn(a, b))
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).unsqueeze_(1)
    indices = list(range(num_train + num_test))
    np.random.shuffle(indices)
    X_train, y_train = X[indices[num_test:]], y[indices[num_test:]]
    X_test, y_test = X[indices[:num_test]], y[indices[:num_test]]
    return X_train, y_train, X_test, y_test


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
        X_train, y_train, X_test, y_test = generate_data(
            num_train=500, num_test=50,
            dim=100, num_sum=5, fn=fn,
            support=RANGE,
        )

        # others
        for net in models:
            model_name = net.__str__().split("(")[0]
            print(f'\tTraining {model_name}...')
            optim = torch.optim.RMSprop(net.parameters(), lr=LEARNING_RATE)
            train(net, optim, X_train, y_train, NUM_ITERS)

            name = f"{fn_str}_{model_name}_trained"
            f = open(f"{save_dir}{name}.obj", "wb")
            pickle.dump(net, f)
            f.close()


if __name__ == '__main__':
    main()
