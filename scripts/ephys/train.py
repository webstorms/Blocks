import sys
import ast
import logging
import argparse

from src import datasets, models, train

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def eval(v):
    return ast.literal_eval(v)


def get_dataset(args):
    if args.downsample == 0.5:
        # Used dt=0.05ms for neural fits
        return datasets.EphysDataset(f"{args.root}/data/ephys", "train", args.neuron_idx, target_sampling_rate=20000)
    else:
        # Used dt>=0.1ms for neural fits (used by most experiments)
        return datasets.EphysDataset(f"{args.root}/data/ephys", "train", args.neuron_idx)


def get_model(args):
    if args.downsample == 0.5:
        # Used dt=0.05ms for neural fits
        return models.Neuron(method=args.method, abs_refac_ms=args.abs_refac_ms, downsample=1, dt01ref=True)
    else:
        # Used dt>=0.1ms for neural fits (used by most experiments)
        return models.Neuron(method=args.method, abs_refac_ms=args.abs_refac_ms, downsample=int(args.downsample))


def get_trainer(args, model, train_dataset):
    n_epochs = 200
    batch_size = 5
    lr = 0.0001
    dt = 0.1 * args.downsample
    return train.EphysTrainer(f"{args.root}/results/ephys/{args.dir_name}", model, train_dataset, n_epochs, batch_size, lr, gamma=0.1, dt=dt, epoch_scan=5, max_decay=0, val_dataset=None, device="cuda", id=args.id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")

    # Model
    parser.add_argument("--method", type=str, default="standard")
    parser.add_argument("--abs_refac_ms", type=int, default=10)
    parser.add_argument("--downsample", type=float, default=1)

    # Dataset
    parser.add_argument("--neuron_idx", type=int, default="")

    # Trainer
    parser.add_argument("--dir_name", type=str, default="")
    parser.add_argument("--id", type=str, default="")

    args = parser.parse_args()

    train_dataset = get_dataset(args)
    model = get_model(args)
    model_trainer = get_trainer(args, model, train_dataset)
    model_trainer.train(save=True)


if __name__ == "__main__":
    main()
