import os
import glob
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor


class ValidNeuronQuery:

    """
    This class builds an idx set of neurons which meet the requirement of having 4 repeats.
    """

    def __init__(self, root):
        self.root = root
        self.query_df = self.build_query_df()

    @property
    def idx(self):
        query = self.query_df["n_trials"] == 4  # Want 4 repeats
        return self.query_df[query]["idx"].values

    def build_query_df(self):
        train_idxs = [v.split("/")[-1] for v in glob.glob(f"{self.root}/train/*")]
        test_idxs = [v.split("/")[-1] for v in glob.glob(f"{self.root}/test/*")]
        joint_idxs = list(set(train_idxs) & set(test_idxs))

        data_list = []

        for idx in joint_idxs:
            try:
                v = torch.load(f"{self.root}/test/{idx}/v.pt")
                n_trials = v.shape[0]
                data_list.append({"idx": idx, "n_trials": n_trials})
            except:
                pass

        return pd.DataFrame(data_list)


class EphysDataset:

    """
    This is the PyTorch friendly e-phys dataset, with all the current and spike tensors.
    """

    LENGTH = 10000  # = 1 second (with DT=0.1)

    def __init__(self, root, dataset, neuron_idx, target_sampling_rate=None):
        # dataset: str which is train or test
        info_df = pd.read_csv(f"{root}/info_df.csv").set_index("idx")

        if target_sampling_rate is None:
            self.i = torch.load(f"{root}/{dataset}/{neuron_idx}/i.pt")[:, :, :1*EphysDataset.LENGTH]
        elif target_sampling_rate == 20000:
            self.i = torch.load(f"{root}/{dataset}/{neuron_idx}/i_20k.pt")[:, :, :2*EphysDataset.LENGTH]
        self.v = torch.load(f"{root}/{dataset}/{neuron_idx}/v.pt")[:, :, :EphysDataset.LENGTH]
        self.s = torch.load(f"{root}/{dataset}/{neuron_idx}/s.pt")[:, :, :EphysDataset.LENGTH]

        vrest = info_df.loc[neuron_idx]["vrest"]
        vthresh = info_df.loc[neuron_idx]["vthresh"]
        self.i = self.i / (100 * self.i.max())
        self.v = (self.v - vrest) / (vthresh - vrest)

    @property
    def hyperparams(self):
        return {}

    def __getitem__(self, item):
        x = self.i[0, item].unsqueeze(0)  # Input current is the same across all trials
        trace = self.v[0, item].unsqueeze(0)
        trace = torch.clamp(trace, -1, 1)
        spikes = self.s[0, item].unsqueeze(0)  # Take first spike trial (all are the same)

        return x, (trace, spikes)

    def __len__(self):
        return 3


class Builder:

    def __init__(self, manifest_file="data/plot_data/allen-brain-observatory/cell_types/manifest.json"):
        self.ctc = CellTypesCache(manifest_file=manifest_file)
        ephys_df = self.generate_ephys_df()
        self.info_df = self.generate_info_df(ephys_df).set_index("idx")

    def save_info_df(self, path):
        self.info_df.to_csv(f"{path}/info_df.csv")

    def build(self, path, **kwargs):
        for neuron_idx in self.info_df.index:
            print(f"Building {neuron_idx}...")
            try:
                meta, i, v = self.generate_all_sweep_tensor(neuron_idx, **kwargs)
                os.mkdir(f"{path}/{neuron_idx}")
                torch.save(i, f"{path}/{neuron_idx}/i.pt")
                torch.save(v, f"{path}/{neuron_idx}/v.pt")
                with open(f"{path}/{neuron_idx}/meta.pkl", "wb") as f:
                    pickle.dump(meta, f)
            except Exception as e:
                print(f"Failed {neuron_idx}: {e}")

    def generate_all_sweep_tensor(self, neuron_idx, target_sampling_rate=10000, start_s=1.02, end_s=1.3):
        sweep_idxs = self.info_df.loc[neuron_idx]["long_square"]
        sweep_dict = {}

        for sweep_idx in sweep_idxs:
            i, v, spike_times = self.generate_sweep_tensor(neuron_idx, sweep_idx, target_sampling_rate, start_s, end_s)
            sweep_dict[i] = (i, v, spike_times)

        # Sort from lowest to highest current
        sweep_dict = {k: v for k, v in sorted(sweep_dict.items(), key=lambda item: item[0])}

        meta = pd.DataFrame([{"i": v[0], "spikes": v[2]} for k, v in sweep_dict.items()])
        v = torch.stack([sweep_dict[key][1] for key in sweep_dict.keys()])

        return meta, v

    def generate_sweep_tensor(self, neuron_idx, sweep_number, target_sampling_rate=10000, start_s=1.02, end_s=1.3):
        data_set = self.ctc.get_ephys_data(neuron_idx)
        sweep_data = data_set.get_sweep(sweep_number)

        index_range = sweep_data["index_range"]
        i = sweep_data["stimulus"][0:index_range[1]+1]  # in A
        v = sweep_data["response"][0:index_range[1]+1]  # in V
        i *= 1e12  # to pA
        v *= 1e3  # to mV

        sampling_rate = int(sweep_data["sampling_rate"])  # in Hz
        t = np.arange(0, len(v)) * (1.0 / sampling_rate)
        downsample_factor = sampling_rate / target_sampling_rate

        sweep_ext = EphysSweepFeatureExtractor(t=t, v=v, i=i, start=start_s, end=end_s)
        sweep_ext.process_spikes()
        spike_times = sweep_ext.spike_feature("threshold_t")

        start_idx = int(start_s*sampling_rate)
        end_idx = int(end_s*sampling_rate)

        downsampled_v, downsampled_i = [], []
        assert v.shape == i.shape

        idx = start_idx
        while idx < end_idx:
            downsampled_v.append(v[int(idx)])
            downsampled_i.append(i[int(idx)])
            idx += downsample_factor

        return torch.Tensor(downsampled_i), torch.Tensor(downsampled_v), spike_times

    def generate_ephys_df(self):
        cells = {cell["id"]: cell for cell in self.ctc.get_cells()}
        ephys_features = self.ctc.get_ephys_features()
        ephys_df = pd.DataFrame(ephys_features)
        ephys_df['id'] = pd.Series([idx for idx in ephys_df['specimen_id']], index=ephys_df.index)
        ephys_df['species'] = pd.Series([cells[idx]['species'] for idx in ephys_df['specimen_id']], index=ephys_df.index)
        ephys_df['dendrite_type'] = pd.Series([cells[idx]['dendrite_type'] for idx in ephys_df['specimen_id']], index=ephys_df.index)
        ephys_df['structure_layer_name'] = pd.Series([cells[idx]['structure_layer_name'] for idx in ephys_df['specimen_id']], index=ephys_df.index)
        ephys_df['disease_state'] = pd.Series([cells[idx]['disease_state'] for idx in ephys_df['specimen_id']], index=ephys_df.index)
        query = ephys_df["structure_layer_name"] == "4"
        query &= ephys_df["species"] == "Mus musculus"
        query &= ephys_df["disease_state"] == ""

        return ephys_df[query]

    def generate_info_df(self, ephys_df):
        info_list = []

        neuron_idxs = ephys_df["id"].values

        for neuron_idx in neuron_idxs:
            # Sweep info
            sweeps = self.ctc.get_ephys_sweeps(neuron_idx)
            sweep_numbers = defaultdict(list)
            for sweep in sweeps:
                sweep_numbers[sweep['stimulus_name']].append(sweep['sweep_number'])

            neuron_type = ephys_df[ephys_df["id"] == neuron_idx]["dendrite_type"].values[0]
            vrest = ephys_df[ephys_df["id"] == neuron_idx]["vrest"].values[0]
            vthresh = ephys_df[ephys_df["id"] == neuron_idx]["threshold_v_long_square"].values[0]

            info_list.append({"idx": neuron_idx, "type": neuron_type, "long_square": sweep_numbers.get("Long Square"), "noise1": sweep_numbers.get("Noise 1"), "noise2": sweep_numbers.get("Noise 2"), "test": sweep_numbers.get("Test"), "vrest": vrest, "vthresh": vthresh})

        return pd.DataFrame(info_list)


class NoiseBuilder(Builder):

    def build(self, path, **kwargs):
        for neuron_idx in self.info_df.index:
            print(f"Building {neuron_idx}...")
            try:
                if not os.path.exists(f"{path}/{neuron_idx}"):
                    os.makedirs(f"{path}/{neuron_idx}")

                i, v, s = self.generate_all_sweep_tensor(neuron_idx, **kwargs)

                # Default used for all experiments
                if kwargs.get("target_sampling_rate") is None:
                    torch.save(i, f"{path}/{neuron_idx}/i.pt")
                    torch.save(v, f"{path}/{neuron_idx}/v.pt")
                    torch.save(s, f"{path}/{neuron_idx}/s.pt")
                # Re-ran some experiments with DT=0.05ms as requested by one reviewer
                elif kwargs.get("target_sampling_rate") == 20000:
                    torch.save(i, f"{path}/{neuron_idx}/i_20k.pt")
                    torch.save(v, f"{path}/{neuron_idx}/v_20k.pt")
                    torch.save(s, f"{path}/{neuron_idx}/s_20k.pt")

            except Exception as e:
                print(f"Failed {neuron_idx}: {e}")

    def generate_all_sweep_tensor(self, neuron_idx, target_sampling_rate=10000, noise_type="noise1"):
        sweep_idxs = self.info_df.loc[neuron_idx][noise_type]
        assert len(sweep_idxs) is not None

        i_list = []
        v_list = []
        s_list = []

        for sweep_idx in sweep_idxs:
            i, v, s = self.generate_noise_sweep_tensor(neuron_idx, sweep_idx, target_sampling_rate)
            i_list.append(i)
            v_list.append(v)
            s_list.append(s)

        return torch.stack(i_list), torch.stack(v_list), torch.stack(s_list)

    def generate_noise_sweep_tensor(self, neuron_idx, sweep_number, target_sampling_rate=10000):
        i1, v1, t1 = self.generate_sweep_tensor(neuron_idx, sweep_number, target_sampling_rate, start_s=2, end_s=5)
        i2, v2, t2 = self.generate_sweep_tensor(neuron_idx, sweep_number, target_sampling_rate, start_s=10, end_s=13)
        i3, v3, t3 = self.generate_sweep_tensor(neuron_idx, sweep_number, target_sampling_rate, start_s=18, end_s=21)
        t1 -= 2
        t2 -= 10
        t3 -= 18
        s1 = NoiseBuilder.to_spike_target(t1, target_sampling_rate)
        s2 = NoiseBuilder.to_spike_target(t2, target_sampling_rate)
        s3 = NoiseBuilder.to_spike_target(t3, target_sampling_rate)

        return torch.stack([i1, i2, i3]), torch.stack([v1, v2, v3]), torch.stack([s1, s2, s3])

    @staticmethod
    def to_spike_target(spike_times, target_sampling_rate):
        dt = 0.0001
        spike_target = torch.zeros(target_sampling_rate)
        spike_idx = [int(spike_time // dt) for spike_time in spike_times if int(spike_time // dt) < target_sampling_rate]
        spike_target[spike_idx] = 1

        return spike_target
