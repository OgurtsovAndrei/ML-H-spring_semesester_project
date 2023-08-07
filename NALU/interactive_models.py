import click
import pickle
import torch

@click.command()
@click.option('-m', '--model', prompt='Choose the model',
              help='Choose the trained model from: MAC, NALU, MLP, MLP_ReLU6')
@click.option('-o', '--operation', prompt='Choose the operation',
              help='Choose the operation from: add, sub, mul, div, root, squared')
@click.option('-v1', '--value1', type=float, prompt='Input float value to compute',
              help='Two float values which will be used in the operation')
@click.option('-v2', '--value2', type=float, prompt='Input float value to compute',
              help='Two float values which will be used in the operation')
def main(model, operation, value1, value2):
    if model not in ['NAC', 'NALU', 'MLP', 'MLP_ReLU6']:
        raise ValueError()
    if operation not in ['add', 'sub', 'mul', 'div', 'root', 'squared']:
        raise ValueError()

    save_dir = 'trained_models/'
    name = f"{operation}_{model}_trained"
    print(f"opening {save_dir}{name}.obj")
    file = open(f"{save_dir}{name}.obj", "rb")
    try:
        trained_model = pickle.load(file)
        print(f"You called model {name} with arguments: {operation} ({value1}, {value2})")
        data = torch.FloatTensor([value1, value2])
        result = trained_model(data)[0]
        print(f"the result is: {result:.3f}")
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()