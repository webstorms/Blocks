import os
import sys
import time
import logging

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from brainbox import trainer
from brainbox.physiology.spiking import VanRossum

from src import datasets, models


torch.backends.cudnn.benchmark = True
logger = logging.getLogger("trainer")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class Trainer(trainer.Trainer):

    def __init__(self, root, model, dataset, n_epochs, batch_size, lr, milestones=[-1], gamma=0.1, val_dataset=None, device="cuda", id=None):
        super().__init__(root, model, dataset, n_epochs, batch_size, lr, torch.optim.Adam, device=device, optimizer_kwargs={"eps": 1e-5}, loader_kwargs={"shuffle": True, "pin_memory": True,  "num_workers": 16}, id=id)
        self._milestones = milestones
        self._gamma = gamma
        self._val_dataset = val_dataset

        self._times = {"forward_pass": [], "backward_pass": []}
        self._train_acc = []
        self._val_acc = []
        self._min_loss = np.inf
        self._milestone_idx = 0

    @staticmethod
    def accuracy_metric(output, target):
        _, predictions = torch.max(output, 1)
        return (predictions == target).sum().cpu().item()

    @staticmethod
    def spike_count(output, target):
        _, cortical_output, thalamic_output = output

        count = cortical_output[0].sum().cpu().item()
        count += thalamic_output[0].sum().cpu().item()

        return count

    @property
    def times_path(self):
        return os.path.join(self.root, self.id, "times.csv")

    @property
    def train_acc_path(self):
        return os.path.join(self.root, self.id, "train_acc.csv")

    @property
    def val_acc_path(self):
        return os.path.join(self.root, self.id, "val_acc.csv")

    def save_model_log(self):
        super().save_model_log()

        # Save times
        times_df = pd.DataFrame(self._times)
        times_df.to_csv(self.times_path, index=False)

        # Save acc
        train_acc_df = pd.DataFrame(self._train_acc)
        train_acc_df.to_csv(self.train_acc_path, index=False)
        val_acc_df = pd.DataFrame(self._val_acc)
        val_acc_df.to_csv(self.val_acc_path, index=False)

    def loss(self, output, target, model):
        target = target.long()
        loss = F.cross_entropy(output, target, reduction="mean")

        return loss

    def train_for_single_epoch(self):
        epoch_loss = 0
        n_samples = 0
        n_correct = 0

        for batch_id, (data, target) in enumerate(self.train_data_loader):
            data = data.to(self.device).type(self.dtype)
            target = target.to(self.device).type(self.dtype)
            torch.cuda.synchronize()

            # Forward pass
            start_time = time.time()
            output = self.model(data)
            torch.cuda.synchronize()
            forward_pass_time = time.time() - start_time
            self._times["forward_pass"].append(forward_pass_time)

            # Compute accuracy
            _, predictions = torch.max(output, 1)
            n_correct += (predictions == target).sum().cpu().item()

            # Compute loss
            loss = self.loss(output, target, self.model)

            # Backward pass
            start_time = time.time()
            loss.backward()
            torch.cuda.synchronize()
            backward_pass_time = time.time() - start_time
            self._times["backward_pass"].append(backward_pass_time)

            self.optimizer.step()
            self.optimizer.zero_grad()

            with torch.no_grad():
                epoch_loss += (loss.item() * data.shape[0])
                n_samples += data.shape[0]

        train_acc = n_correct/n_samples
        logging.info(f"Train acc: {train_acc}")
        self._train_acc.append(train_acc)

        if self._val_dataset is not None and len(self.log["train_loss"]) % 5 == 0:
            val_acc = Trainer.get_acc(self.model, self._val_dataset, self.batch_size)
            logging.info(f"Val acc: {val_acc}")
            self._val_acc.append(val_acc)

        return epoch_loss / n_samples

    @staticmethod
    def get_acc(model, dataset, batch_size):
        scores = trainer.compute_metric(model, dataset, Trainer.accuracy_metric, batch_size=batch_size)
        return np.sum(scores) / len(dataset)

    @staticmethod
    def get_spike_count(model, dataset, batch_size):
        scores = trainer.compute_metric(model, dataset, Trainer.spike_count, batch_size=batch_size)
        return np.sum(scores) / len(dataset)

    def on_epoch_complete(self, save):
        if save:
            self.save_model_log()

            epoch_loss = self.log["train_loss"][-1]
            if epoch_loss < self._min_loss:
                logging.info(f"Saving model...")
                self._min_loss = epoch_loss
                self.save_model()

        n_epoch = len(self.log["train_loss"])

        if n_epoch == self._milestones[self._milestone_idx]:
            logging.info(f"Decaying lr...")
            self.lr *= self._gamma
            # Load best model
            self.model = Trainer.load_model(self.root, self.id, self.device, self.dtype)
            self.optimizer = self.optimizer_func(
                self.model.parameters(), self.lr, **self.optimizer_kwargs
            )

            if self._milestone_idx != len(self._milestones) - 1:
                logging.info(f"New milestone target...")
                self._milestone_idx += 1

    def on_training_complete(self, save):
        pass

    @staticmethod
    def hyperparams_loader(hyperparams):
        model_params = hyperparams["model"]
        del model_params["name"]
        del model_params["weight_initializers"]

        return models.AuditoryModel(**model_params)

    @staticmethod
    def load_model(root, id, device="cuda", dtype=torch.float):
        return trainer.load_model(root, id, Trainer.hyperparams_loader, device, dtype)


class EphysTrainer(Trainer):

    def __init__(self, root, model, dataset, n_epochs, batch_size, lr, gamma=0.1, dt=0.1, epoch_scan=5, max_decay=1, val_dataset=None, device="cuda", id=None):
        super().__init__(root, model, dataset, n_epochs, batch_size, lr, [-1], gamma, val_dataset, device, id)
        self.epoch_scan = epoch_scan
        self.van_rossum = VanRossum(datasets.EphysDataset.LENGTH, tau=100, dt=dt).to(device)
        self.max_decay = max_decay

        self._decay_count = 0

    def loss(self, spikes_pred, spikes):
        spike_loss = self.van_rossum(spikes_pred, spikes)

        return spike_loss

    def train_for_single_epoch(self):
        epoch_loss = 0
        n_samples = 0

        for batch_id, (data, target) in enumerate(self.train_data_loader):
            data = data.to(self.device).type(self.dtype)
            trace = target[0].to(self.device).type(self.dtype)
            spikes = target[1].to(self.device).type(self.dtype)
            torch.cuda.synchronize()

            # Forward pass
            start_time = time.time()
            spikes_pred = self.model(data)
            torch.cuda.synchronize()
            forward_pass_time = time.time() - start_time
            self._times["forward_pass"].append(forward_pass_time)

            # Compute loss
            loss = self.loss(spikes_pred, spikes)

            # Backward pass
            start_time = time.time()
            loss.backward()
            torch.cuda.synchronize()
            backward_pass_time = time.time() - start_time
            self._times["backward_pass"].append(backward_pass_time)

            self.optimizer.step()
            self.optimizer.zero_grad()

            with torch.no_grad():
                epoch_loss += (loss.item() * data.shape[0])
                n_samples += data.shape[0]

        return epoch_loss / n_samples

    def on_epoch_complete(self, save):
        if save:
            self.save_model_log()

            epoch_loss = self.log["train_loss"][-1]
            if epoch_loss < self._min_loss:
                logging.info(f"Saving model...")
                self._min_loss = epoch_loss
                self.save_model()

        min_lost_over_last_epochs = np.array(self.log["train_loss"][-self.epoch_scan:]).min()

        if min_lost_over_last_epochs > self._min_loss:
            if self._decay_count < self.max_decay:
                self._min_loss = np.inf
                self._last_train_scores = []
                logging.info(f"Decaying lr...")
                self._decay_count += 1
                self.lr *= self._gamma
                # Load best model
                self.model = EphysTrainer.load_model(self.root, self.id, self.device, self.dtype)
                self.optimizer = self.optimizer_func(
                    self.model.parameters(), self.lr, **self.optimizer_kwargs
                )
            else:
                self.exit = True

    @staticmethod
    def load_model(root, id, device="cuda", dtype=torch.float, dt01ref=False):

        def model_loader(hyperparams):
            model_params = hyperparams["model"]
            del model_params["name"]
            del model_params["weight_initializers"]

            model_params = {**model_params, "dt01ref": dt01ref}

            return models.Neuron(**model_params)

        return trainer.load_model(root, id, model_loader, device, dtype)
