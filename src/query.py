import glob

import pandas as pd
from brainbox import trainer

from src import models, train


models.snn.BaseSNN.MIN_BETA = 0.01
models.snn.BaseSNN.MAX_BETA = 0.99


class BenchmarkQuery:

    def __init__(self, root, batches=[32, 64, 128]):
        self._root = root

        self._results_df = self._build_df()
        self._results_df = pd.concat([self._query_results(batch=b) for b in batches])

    def _build_df(self):
        results_df_list = []

        for path in self._get_paths(self._root):
            results_df_list.append(pd.read_csv(path))

        results_df = pd.concat(results_df_list)
        results_df["total_time"] = results_df["forward_time"] + results_df["backward_time"]

        return results_df

    def _get_paths(self, root):
        return [path for path in glob.glob(f"{root}/*")]

    def _query_results(self, **kwargs):
        query = True
        for key, value in kwargs.items():
            query &= self._results_df[key] == value

        if len(kwargs) > 0:
            return self._results_df[query]

        return self._results_df

    def get_speedup(self):
        results_df = self._build_df()
        standard_times = results_df[results_df["method"] == "standard"].set_index(["t_len", "units", "batch", "abs_refac", "layers"])[["forward_time", "backward_time", "total_time"]]
        blocks_times = results_df[results_df["method"] == "blocks"].set_index(["t_len", "units", "batch", "abs_refac", "layers"])[["forward_time", "backward_time", "total_time"]]

        speedup_df = standard_times / blocks_times
        speedup_df.rename(columns={"forward_time": "forward_speedup", "backward_time": "backward_speedup", "total_time": "total_speedup"}, inplace=True)

        return speedup_df


class SupervisedQuery:

    def __init__(self, root):
        self.root = root

    def get_average_duration_per_batch(self, models_root, model_id):
        durations_list = []

        duration = trainer.load_log(models_root, model_id)["duration"][1:].mean()
        durations_list.append({"model_id": model_id, "duration": duration})

        return pd.DataFrame(durations_list).set_index("model_id").values[0][0]

    def build_results(self, dataset, methods, sgs, abs_refacs, repeats, detach=True, batch_size=500):
        results_list = []

        for method in methods:
            for sg in sgs:
                for abs_refac in abs_refacs:
                    for i in range(repeats):
                        if detach:
                            name = f"{dataset.name}_{method}_{sg}_{abs_refac}_{dataset.dt}_{i}"
                        else:
                            name = f"{dataset.name}_{method}_{sg}_{abs_refac}_{dataset.dt}_{detach}_{i}"
                        print(f"Loading {name}...")
                        model = train.Trainer.load_model(f"{self.root}/results/supervised", name)
                        val_acc = train.Trainer.get_acc(model, dataset, batch_size)
                        avg_time = self.get_average_duration_per_batch(f"{self.root}/results/supervised", name)
                        results_list.append({"dataset": dataset.name, "method": method, "sg": sg, "abs_refac": abs_refac, "dt": dataset.dt, "i": i, "val_acc": val_acc, "avg_time": avg_time})

        return pd.DataFrame(results_list)