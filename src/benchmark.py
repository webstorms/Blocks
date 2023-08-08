import os
import time

import torch
import pandas as pd

from src import datasets
from src import models


class Benchmarker:

    def __init__(self, method, t_len, abs_refac, n_in, n_hidden, n_layers, batch_size=16, min_r=0, max_r=200, n_samples=11):
        self._method = method
        self._t_len = t_len
        self._abs_refac = abs_refac
        self._n_in = n_in
        self._n_hidden = n_hidden
        self._n_layers = n_layers
        self._batch_size = batch_size
        self._model = models.ModelBuilder(method, t_len, abs_refac, n_in, n_hidden, n_layers)

        self._data_loader = self._get_data_loader(t_len, n_in, min_r, max_r, batch_size, n_samples*batch_size)
        self._benchmark_results = None

    def benchmark(self, device="cuda"):
        timing_list = []

        self._model = self._model.to(device)

        for i, data in enumerate(self._data_loader):
            # Benchmark forward pass
            data = data.to(device)

            start_time = time.time()
            output = self._model(data)
            torch.cuda.synchronize()
            forward_pass_time = time.time() - start_time

            # Benchmark backward pass
            start_time = time.time()
            loss = output.sum()
            loss.backward()
            torch.cuda.synchronize()
            backward_pass_time = time.time() - start_time

            # Ignore first run (as this usually loads things which slows things down)
            # e.g. cudnn finds best conv algorithm
            if i > 0:
                timing_row = {"forward_time": forward_pass_time, "backward_time": backward_pass_time}
                timing_list.append(timing_row)

        self._benchmark_results = timing_list

    def save(self, path):
        results_df = self._to_df()
        results_df.to_csv(os.path.join(path, f"{self._get_df_name()}.csv"), index=False)

    def _get_description(self):
        return {"method": self._method, "t_len": self._t_len, "abs_refac": self._abs_refac, "units": self._n_hidden, "layers": self._n_layers, "batch": self._batch_size}

    def _get_df_name(self):
        return f"{self._method}_{self._t_len}_{self._abs_refac}_{self._n_hidden}_{self._n_layers}_{self._batch_size}"

    def _get_data_loader(self, t_len, n_units, min_r, max_r, batch_size, n_samples):
        spikes_dataset = datasets.SyntheticSpikes(t_len, n_units, min_r, max_r, n_samples)
        return torch.utils.data.DataLoader(spikes_dataset, batch_size, shuffle=False)

    def _to_df(self):
        assert self._benchmark_results is not None
        results = []

        for results_row in self._benchmark_results:
            results.append({**results_row, **self._get_description()})

        return pd.DataFrame(results)
