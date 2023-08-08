import sys
import ast
import logging
import argparse

from src import datasets, models, train

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def eval(v):
    return ast.literal_eval(v)


def get_dataset(args, train=True):
    if args.name == "shd":
        return datasets.SHDDataset(f"{args.root}/data/SHD", train=train, dt=args.dt)
    elif args.name == "nmnist":
        return datasets.NMNISTDataset(f"{args.root}/data/N-MNIST", train=train, dt=args.dt)


def get_model(args, dataset):
    abs_refac = int(args.abs_refac / args.dt)

    if args.name == "shd":
        n_in = 700
        n_out = 20
    elif args.name == "nmnist":
        n_in = 1156
        n_out = 10

    return models.AuditoryModel(args.method, n_in, args.n_hidden, n_out, dataset.t_len, abs_refac, eval(args.recurrent), args.dt, args.surr_grad, detach_spike_grad=eval(args.detach_spike_grad))


def get_trainer(args, model, train_dataset, val_dataset):
    gamma = 0.1
    if args.name == "shd":
        milestones = [15, 15]
        epochs = 40
    elif args.name == "nmnist":
        milestones = [30]
        epochs = 20
    return train.Trainer(f"{args.root}/results", model, train_dataset, epochs, args.batch_size, args.lr, milestones, gamma, val_dataset, id=args.id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")

    # Model
    parser.add_argument("--method", type=str, default="standard")
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--abs_refac", type=float, default=10)
    parser.add_argument("--recurrent", type=str, default="True")
    parser.add_argument("--detach_spike_grad", type=str, default="True")
    parser.add_argument("--surr_grad", type=str, default="fast_sigmoid")

    # Dataset
    parser.add_argument("--name", type=str, default="shd")
    parser.add_argument("--dt", type=float, default=1)

    # Trainer
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument('--id', type=str, default="")

    args = parser.parse_args()

    train_dataset = get_dataset(args, train=True)
    val_dataset = get_dataset(args, train=False)
    if args.name == "nmnist":
        val_dataset = None
    model = get_model(args, train_dataset)
    model_trainer = get_trainer(args, model, train_dataset, val_dataset)
    model_trainer.train(save=True)


if __name__ == "__main__":
    main()
