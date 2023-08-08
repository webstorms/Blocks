import torch
import pandas as pd
from brainbox.physiology.spiking import SpikeToPSTH

from src import datasets, train


class SpikeTrainEV:

    def __init__(self, st_len, sig):
        self._spike_smoother = SpikeToPSTH(st_len, sig)

    def __call__(self, input, target):
        # input: b x 3 x t
        smooth_input_trains = self.smooth_spike_trains(input)
        smooth_target_trains = self.smooth_spike_trains(target)
        flatten_input_trains = self.flatten_spike_trains(smooth_input_trains)
        flatten_target_trains = self.flatten_spike_trains(smooth_target_trains)

        return SpikeTrainEV.ev(flatten_input_trains, flatten_target_trains.mean(0).unsqueeze(0)).mean()

    def smooth_spike_trains(self, spike_trains):
        # spike_trains: b x 3 x t
        b = spike_trains.shape[0]

        return self._spike_smoother(spike_trains.view(b*3, -1).unsqueeze(1))[:, 0].view(b, 3, -1)

    def flatten_spike_trains(self, spike_trains):
        # spike_trains: b x 3 x t
        return spike_trains.flatten(1, 2)

    @staticmethod
    def ev(input, target):
        # input: b x t
        return (input.var(1) + target.var(1) - (input - target).var(1)) / (input.var(1) + target.var(1))


class EphysAnalysis:

    def __init__(self, root, method, abs_refac_ms, downsample, taus=[100]):
        self.root = root
        self.method = method
        self.abs_refac_ms = abs_refac_ms
        self.downsample = downsample
        self.taus = taus

        self.metrics = {tau: SpikeTrainEV(10000, tau) for tau in taus}
        self.neuron_idxs = datasets.ValidNeuronQuery(f"{root}/data/ephys").idx

        # Init test dataset
        self.test_dataset = {}
        self.norm_factors = {tau: {} for tau in taus}

        for neuron_idx in self.neuron_idxs:
            if downsample == 0.5:
                self.test_dataset[neuron_idx] = datasets.EphysDataset(f"{root}/data/ephys", "test", int(neuron_idx), target_sampling_rate=20000)
            else:
                self.test_dataset[neuron_idx] = datasets.EphysDataset(f"{root}/data/ephys", "test", int(neuron_idx))
            target_spikes = self.test_dataset[neuron_idx].s.cpu()
            for tau in taus:
                self.norm_factors[tau][neuron_idx] = self.metrics[tau](target_spikes, target_spikes).item()

        self._norm_df = {tau: pd.Series(self.norm_factors[tau]).to_frame().rename(columns={0: "score"}) for tau in taus}
        self._ev_df = None

    def ev_df(self, tau, normalise):
        if self._ev_df is None:
            self._ev_df = self._build_ev_df().set_index("neuron_idx")

        query = self._ev_df["tau"] == tau
        ev_df = self._ev_df[query]["score"].to_frame()

        if normalise:
            df = ev_df / self._norm_df[tau]
        else:
            df = ev_df

        df.index = df.index.map(int)

        return df

    def get_times_df(self):
        dir_name = f"{self.method}_{self.abs_refac_ms}_{self.downsample}"

        times_list = []

        for neuron_idx in self.neuron_idxs:
            times_csv = pd.read_csv(f"{self.root}/results/ephys/{dir_name}/{neuron_idx}/times.csv")
            forward_pass, backward_pass = times_csv.sum()
            times_list.append({"neuron_idx": neuron_idx, "forward_pass": forward_pass, "backward_pass": backward_pass})

        return pd.DataFrame(times_list)

    def _build_ev_df(self):
        dir_name = f"{self.method}_{self.abs_refac_ms}_{self.downsample}"

        metric_list = []

        for i, neuron_idx in enumerate(self.neuron_idxs):
            print(f"Building {i}/{len(self.neuron_idxs)} {neuron_idx}...")
            dt01ref = self.downsample == 0.5
            neuron = train.EphysTrainer.load_model(f"{self.root}/results/ephys/{dir_name}", neuron_idx, dt01ref=dt01ref)

            with torch.no_grad():
                test_dataset = self.test_dataset[neuron_idx]
                pred_spikes = neuron(test_dataset.i[0].unsqueeze(1).cuda()).permute(1, 0, 2).cpu()
                target_spikes = test_dataset.s.cpu()

                for tau in self.taus:
                    score = self.metrics[tau](target_spikes, pred_spikes).item()  # note: model spikes are reference
                    metric_list.append({"neuron_idx": neuron_idx, "score": score, "tau": tau})

        return pd.DataFrame(metric_list)

    def load_prediction(self, neuron_idx):  # Load prediction for a certain fit and neuron
        test_dataset = self.test_dataset[str(neuron_idx)]

        with torch.no_grad():
            dir_name = f"{self.method}_{self.abs_refac_ms}_{self.downsample}"
            dt01ref = self.downsample == 0.5
            neuron = train.EphysTrainer.load_model(f"{self.root}/results/ephys/{dir_name}", neuron_idx, dt01ref=dt01ref)

            output = neuron(test_dataset.i[0].unsqueeze(1).cuda(), mode="val")
            spikes = output[0].permute(1, 0, 2).cpu()
            mem = output[1].permute(1, 0, 2).cpu()

            return spikes, mem, test_dataset.s, test_dataset.v, test_dataset.i

    def load_params_df(self):
        dir_name = f"{self.method}_{self.abs_refac_ms}_{self.downsample}"

        params_list = []

        for i, neuron_idx in enumerate(self.neuron_idxs):
            print(f"Building {i}/{len(self.neuron_idxs)} {neuron_idx}...")
            neuron = train.EphysTrainer.load_model(f"{self.root}/results/ephys/{dir_name}", neuron_idx).neuron

            params_list.append({"neuron_idx": neuron_idx, "beta": neuron.beta.item(), "p": neuron.p.item(), "b": neuron.b.item()})

        return pd.DataFrame(params_list)
