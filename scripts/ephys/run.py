import os
root = ""  # TODO: Change this to the project folder

from src import datasets


def launch(method, abs_refac_ms, downsample):
    neuron_idx_query = datasets.ValidNeuronQuery(f"{root}/data/ephys")

    dir_name = f"{method}_{abs_refac_ms}_{downsample}"
    os.makedirs(f"{root}/results/ephys/{dir_name}")

    for neuron_idx in neuron_idx_query.idx:
        os.system(f"python {root}/scripts/ephys/train.py --root={root} --method={method} --abs_refac_ms={abs_refac_ms} --downsample={downsample} --neuron_idx={neuron_idx} --dir_name={dir_name} --id={neuron_idx}")


# Blocks: Different dt
for downsample in [0.5, 1, 5, 10, 20, 40]:
    launch("blocks", abs_refac_ms=2, downsample=downsample)

# Blocks: Different ARP
for abs_refac_ms in [1, 4, 6, 8, 16]:
    launch("blocks", abs_refac_ms=abs_refac_ms, downsample=1)

# Standard
launch("standard", abs_refac_ms=2, downsample=1)
