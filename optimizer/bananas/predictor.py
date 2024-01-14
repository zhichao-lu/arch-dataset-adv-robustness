import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from collections import defaultdict
from utils import AverageMeter

import numpy as np

class FeedforwardNet(nn.Module):
    def __init__(
        self,
        input_dims: int = 5,
        num_layers: int = 3,
        layer_width: list = [10, 10, 10],
        output_dims: int = 1,
        activation="relu",
    ):
        super(FeedforwardNet, self).__init__()
        assert (
            len(layer_width) == num_layers
        ), "number of widths should be \
        equal to the number of layers"

        self.activation = eval("F." + activation)

        all_units = [input_dims] + layer_width
        self.layers = nn.ModuleList(
            [nn.Linear(all_units[i], all_units[i + 1]) for i in range(num_layers)]
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
        return self.out(x)

    def basis_funcs(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

class MLPPredictor():
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

        self.model = self.get_model(
            input_dims=xtrain.shape[1],
            num_layers=self.num_layers,
            layer_width=[self.layer_width]*self.num_layers,
        )
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.99))

        if self.loss_type == "mse":
            criterion = nn.MSELoss().to(self.device)
        elif self.loss_type == "mae":
            criterion = nn.L1Loss().to(self.device)

        self.model.train()

        for e in range(self.epochs):
            meters = defaultdict(AverageMeter)
            for input,target in data_loader:
                input = input.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                prediction = self.model(input).view(-1)

                loss_fn = criterion(prediction, target)
                # add L1 regularization
                params = torch.cat(
                    [
                        x[1].view(-1)
                        for x in self.model.named_parameters()
                        if x[0] == "out.weight"
                    ]
                )
                loss_fn += self.l1_ref * torch.norm(params, 1)
                loss_fn.backward()
                optimizer.step()

                mse = accuracy_mse(prediction.detach(), target)
                meters.update(
                    {"loss": loss_fn.item(), "mse": mse.item()}, n=target.size(0)
                )

            if verbose and e % 100 == 0:
                print("Epoch {}, {}, {}".format(e, meters["loss"], meters["mse"]))

        train_pred = self.query(xtrain)
        train_error = np.mean(np.abs(train_pred - ytrain))
        return train_error

    def query(self, xtest, eval_batch_size=None):

        test_data = TensorDataset(xtest)

        eval_batch_size = len(xtest) if eval_batch_size is None else eval_batch_size
        test_data_loader = DataLoader(
            test_data, batch_size=eval_batch_size, pin_memory=False
        )

        self.model.eval()
        pred = []
        with torch.no_grad():
            for batch in test_data_loader:
                input = batch[0].to(self.device)
                prediction = self.model(input)
                pred.append(prediction)

        return torch.cat(pred).cpu().numpy()



def accuracy_mse(prediction, target, scale=100.0):
    return F.mse_loss(prediction, target) * scale



class Ensemble:
    def __init__(self, cfg):
        self.num_ensemble = cfg.num_ensemble
        self.cfg = cfg

        self.ensamble = [None] * self.num_ensemble
    

    def fit(self, xtrain, ytrain, verbose=False):
        train_errors = []
        for i in range(self.num_ensemble):
            self.ensamble[i] = MLPPredictor(self.cfg)
            train_error = self.ensamble[i].fit(xtrain, ytrain, verbose=verbose)
            train_errors.append(train_error)

        return train_errors
    
    def query(self, xtest, eval_batch_size=None):
        predictions = []
        for i in range(self.num_ensemble):
            predictions.append(self.ensamble[i].query(xtest, eval_batch_size=eval_batch_size))
        return predictions
            