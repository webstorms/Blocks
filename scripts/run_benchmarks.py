import torch
torch.backends.cudnn.benchmark = True  # Important in order to use best conv algorithm

from src.benchmark import Benchmarker


def run_different_sim_lengths(root):
    n_in = 1000
    n_hidden = 128

    for abs_refac in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        for t_len in [2**9, 2**10, 2**11]:
            for batch_size in [32, 64, 128]:
                for method in ["standard", "blocks"]:
                    bencher = Benchmarker(method, t_len, abs_refac, n_in, n_hidden, n_layers=1, batch_size=batch_size)
                    bencher.benchmark()
                    bencher.save(root)


def run_different_layers(root):
    n_in = 1000

    for abs_refac in [40]:
        for t_len in [2**10]:
            for n_hidden in [128, 256, 512]:
                for batch_size in [64]:
                    for n_layers in [2, 3, 4, 5]:
                        for method in ["standard", "blocks"]:
                            bencher = Benchmarker(method, t_len, abs_refac, n_in, n_hidden, n_layers=n_layers, batch_size=batch_size)
                            bencher.benchmark()
                            bencher.save(root)


if __name__ == "__main__":
    root = ""  # TODO: Change this to the project folder
    run_different_sim_lengths(f"{root}/benchmarks/sim_lengths")
    run_different_layers(f"{root}/benchmarks/layers")

