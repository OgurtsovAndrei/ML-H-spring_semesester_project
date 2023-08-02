import pickle
import torch

import random
import numpy as np

import torch
import torch.nn.functional as F

from models import MLP, NAC, NALU
from genereate_data import generate_data

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


def test(model, data, target):
    with torch.no_grad():
        out = model(data)
        return torch.abs(target - out)


def main():
    save_dir = './results/'
    models_dir = './trained_models/'

    model_names = ['NAC', 'NALU', 'MLP', 'MLP_ReLU6']

    results = {}
    for operation, fn in ARITHMETIC_FUNCTIONS.items():
        print(f'[*] Testing operation: {operation}')
        results[operation] = []

        # dataset
        X_test, y_test = generate_data(
            num_data=500,
            dim=100, num_sum=5, fn=fn,
            support=RANGE,
        )

        # random model
        random_mse = []
        for i in range(100):
            net = MLP(
                num_layers=NUM_LAYERS, in_dim=2,
                hidden_dim=HIDDEN_DIM, out_dim=1,
                activation='relu6',
            )
            mse = test(net, X_test, y_test)
            random_mse.append(mse.mean().item())
        results[operation].append(np.mean(random_mse))

        # others
        for model_name in model_names:
            print(f'\tTesting {model_name}...')
            name = f"{operation}_{model_name}_trained"
            print(f"opening {models_dir}{name}.obj")
            file = open(f"{models_dir}{name}.obj", "rb")
            try:
                trained_model = pickle.load(file)
                mse = test(trained_model, X_test, y_test).mean().item()
                results[operation].append(mse)
            except Exception as e:
                print(e)

    with open(save_dir + "interpolation.txt", "w") as f:
        f.write("Relu6\tNone\tNAC\tNALU\n")
        for k, v in results.items():
            rand = results[k][0]
            mses = [100.0*x/rand for x in results[k][1:]]
            if NORMALIZE:
                f.write("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(*mses))
            else:
                f.write("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(*results[k][1:]))

    print("done")

if __name__ == '__main__':
    main()
