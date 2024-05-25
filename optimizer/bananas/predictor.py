import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from collections import defaultdict
from utils import AverageMeter

# from concurrent.futures import ThreadPoolExecutor, as_completed

# import numpy as np


def mape_loss(prediction, target, target_lb):
    fraction = (prediction-target_lb)/(target-target_lb)
    return torch.abs(fraction-1)


class FeedforwardNet(nn.Module):
    def __init__(
        self,
        input_dims: int = 5,
        num_layers: int = 3,
        layer_width: list = [10, 10, 10],
        activation=F.relu,
    ):
        super(FeedforwardNet, self).__init__()
        assert (
            len(layer_width) == num_layers
        ), "number of widths should be \
        equal to the number of layers"

        self.activation = activation

        all_units = [input_dims] + layer_width
        self.layers = nn.ModuleList(
            [nn.Linear(all_units[i], all_units[i + 1])
             for i in range(num_layers)]
        )

        self.out = nn.Linear(all_units[-1], 1)

        # make the init similar to the tf.keras version
        for l in self.layers:
            torch.nn.init.xavier_uniform_(l.weight)
            torch.nn.init.zeros_(l.bias)
        torch.nn.init.xavier_uniform_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.out(x).squeeze(1)


class MLPPredictor:
    def __init__(self, cfg, device=torch.device("cpu")):

        self.num_layers = cfg.num_layers
        self.layer_width = cfg.layer_width
        self.batch_size = cfg.batch_size
        self.lr = cfg.lr
        self.l1_reg = cfg.l1_reg
        self.epochs = cfg.epochs
        self.loss_type = cfg.loss_type

        self.model = None
        self.device = device

    def get_model(self, **kwargs):
        predictor = FeedforwardNet(**kwargs)
        return predictor

    def fit(self, xtrain, ytrain, verbose=False):

        # self.mean = np.mean(ytrain)
        # self.std = np.std(ytrain)

        train_data = TensorDataset(xtrain, ytrain)
        data_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=False,
        )

        y_lb = ytrain.min()

        self.model = self.get_model(
            input_dims=xtrain.shape[1],
            num_layers=self.num_layers,
            layer_width=[self.layer_width] * self.num_layers,
        )
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.lr, betas=(0.9, 0.99))
        
        self.model.train()

        for e in range(self.epochs):
            meters = defaultdict(AverageMeter)
            for input, target in data_loader:
                input = input.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                prediction = self.model(input)

                # loss = criterion(prediction, target)
                if self.loss_type == "mse":
                    loss = F.mse_loss(prediction, target)
                elif self.loss_type == "mae":
                    loss = F.l1_loss(prediction, target)
                elif self.loss_type == 'mape':
                    loss = mape_loss(prediction, target, y_lb)
                else:
                    raise ValueError(f'Loss Type "{self.loss_type}" is not supported')

                # add L1 regularization on output layer
                params = torch.cat(
                    [
                        x[1].view(-1)
                        for x in self.model.named_parameters()
                        if x[0] == "out.weight"
                    ]
                )
                loss += self.l1_reg * torch.norm(params, 1)
                loss.backward()
                optimizer.step()

                mse = accuracy_mse(prediction.detach(), target)

                meters["loss"].update(loss.item(), n=target.size(0))
                meters['mse'].update(mse.item(), n=target.size(0))

            if verbose and e % 100 == 0:
                print(f'Epoch {e}, loss: {meters["loss"]}, {meters["mse"]}')

        train_pred = self.query(xtrain)
        train_error = torch.mean(torch.abs(train_pred - ytrain))
        return train_error

    def query(self, xtest, eval_batch_size=None):
        test_data = TensorDataset(xtest)
        eval_batch_size = xtest.shape[0] if eval_batch_size is None else eval_batch_size

        test_data_loader = DataLoader(
            test_data, batch_size=eval_batch_size, pin_memory=False
        )

        self.model.eval()
        pred = []
        with torch.no_grad():
            for batch in test_data_loader:
                x = batch[0].to(self.device)
                prediction = self.model(x)
                pred.append(prediction)

        return torch.cat(pred)


def accuracy_mse(prediction, target, scale=100.0):
    return F.mse_loss(prediction, target) * scale


class Ensemble:
    def __init__(self, cfg, device=torch.device("cpu")):
        self.num_ensemble = cfg.num_ensemble
        self.cfg = cfg

        self.device = device

        self.ensamble = [None] * self.num_ensemble

    def fit(self, xtrain, ytrain, verbose=True):
        xtrain = torch.from_numpy(xtrain).to(self.device, dtype=torch.float32)
        ytrain = torch.from_numpy(ytrain).to(self.device, dtype=torch.float32)

        train_errors = []
        for i in range(self.num_ensemble):
            self.ensamble[i] = MLPPredictor(self.cfg, self.device)
            train_errors.append(
                self.ensamble[i]
                .fit(xtrain, ytrain, verbose=verbose)
                .detach()
                .cpu()
                .numpy()
            )

        # with ThreadPoolExecutor(max_workers=self.num_ensemble) as thread_pool:
        #     tasks = []
        #     for i in range(self.num_ensemble):
        #         self.ensamble[i] = MLPPredictor(self.cfg, self.device)
        #         tasks.append(
        #             thread_pool.submit(self.ensamble[i].fit, xtrain, ytrain, verbose=verbose)
        #         )

        #         # train_errors.append(
        #         #     self.ensamble[i]
        #         #     .fit(xtrain, ytrain, verbose=verbose)
        #         #     .detach()
        #         #     .cpu()
        #         #     .numpy()
        #         # )
        #     for future in as_completed(tasks):
        #         train_errors.append(future.result().detach().cpu().numpy())

        return train_errors

    def query(self, xtest, eval_batch_size=None):
        xtest = torch.from_numpy(xtest).to(self.device, dtype=torch.float32)
        eval_batch_size = xtest.shape[0] if eval_batch_size is None else eval_batch_size

        predictions = []
        for i in range(self.num_ensemble):
            predictions.append(
                self.ensamble[i]
                .query(xtest, eval_batch_size=eval_batch_size)
                .detach()
                .cpu()
                .numpy()
            )
        return predictions
