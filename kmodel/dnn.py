"""Deep learning helpers built on the existing fastai training backbone, with runnable examples from a seaborn dataset."""


import os
import random

import fastcore.all as fc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision.all import *
from torch.utils.data import DataLoader, Dataset
from pathlib import Path


def_device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

class GeneralDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feat_col,
        target_col=None,
        A: int = 23,
        dtype=np.float32,
    ) -> None:
        "Return features only in test mode, or (features, targets) in training mode." 
        self.test = target_col is None
        self.aa = A
        self.X = df[list(feat_col)].to_numpy(dtype=dtype, copy=True)
        self.y = None
        if not self.test:
            y_flat = df[list(target_col)].to_numpy(dtype=dtype, copy=True)
            total = y_flat.shape[1]
            if total % A != 0:
                raise ValueError(f"Target columns ({total}) not divisible by A={A}; cannot infer positions.")
            self.position = total // self.aa
            self.y = y_flat.reshape(-1, A, self.position)
        self.len = len(df)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        X = torch.from_numpy(self.X[index])
        if self.test:
            return X
        y = torch.from_numpy(self.y[index])
        return X, y

def MLP(
    num_features: int,
    num_targets: int,
    hidden_units: list[int] = [512, 218],
    dp: float = 0.2,
):
    "Feed-forward model for tabular inputs." 
    layers = [
        nn.Linear(num_features, hidden_units[0]),
        nn.BatchNorm1d(hidden_units[0]),
        nn.Dropout(dp),
        nn.PReLU(),
    ]
    for i in range(len(hidden_units) - 1):
        layers.extend([
            nn.Linear(hidden_units[i], hidden_units[i + 1]),
            nn.BatchNorm1d(hidden_units[i + 1]),
            nn.Dropout(dp),
            nn.PReLU(),
        ])
    layers.append(nn.Linear(hidden_units[-1], num_targets))
    return nn.Sequential(*layers)

def lin_wn(ni, nf, dp: float = 0.1, act=nn.SiLU):
    "Weight-normalized linear block." 
    layers = [
        nn.BatchNorm1d(ni),
        nn.Dropout(dp),
        nn.utils.parametrizations.weight_norm(nn.Linear(ni, nf)),
    ]
    if act:
        layers.append(act())
    return nn.Sequential(*layers)

def conv_wn(ni, nf, ks: int = 3, stride: int = 1, padding: int = 1, dp: float = 0.1, act=nn.ReLU):
    "Weight-normalized convolution block." 
    layers = [
        nn.BatchNorm1d(ni),
        nn.Dropout(dp),
        nn.utils.parametrizations.weight_norm(nn.Conv1d(ni, nf, ks, stride, padding)),
    ]
    if act:
        layers.append(act())
    return nn.Sequential(*layers)

class CNN1D(nn.Module):
    def __init__(self, ni, nf, amp_scale: int = 16):
        super().__init__()
        cha_1, cha_2, cha_3 = 256, 512, 512
        hidden_size = cha_1 * amp_scale
        cha_po_1 = hidden_size // (cha_1 * 2)
        cha_po_2 = (hidden_size // (cha_1 * 4)) * cha_3

        self.lin = lin_wn(ni, hidden_size)
        self.view = View(-1, cha_1, amp_scale)
        self.conv1 = nn.Sequential(
            conv_wn(cha_1, cha_2, ks=5, stride=1, padding=2, dp=0.1),
            nn.AdaptiveAvgPool1d(output_size=cha_po_1),
            conv_wn(cha_2, cha_2, ks=3, stride=1, padding=1, dp=0.1),
        )
        self.conv2 = nn.Sequential(
            conv_wn(cha_2, cha_2, ks=3, stride=1, padding=1, dp=0.3),
            conv_wn(cha_2, cha_3, ks=5, stride=1, padding=2, dp=0.2),
        )
        self.head = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),
            nn.Flatten(),
            lin_wn(cha_po_2, nf, act=None),
        )

    def forward(self, x):
        x = self.lin(x)
        x = self.view(x)
        x = self.conv1(x)
        x_s = x
        x = self.conv2(x)
        x = x * x_s
        return self.head(x)

def init_weights(m, leaky: float = 0.0):
    "Initialize convolution layers with Kaiming normal weights." 
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(m.weight, a=leaky)

class PSSM_model(nn.Module):
    def __init__(self, n_features, n_targets, A: int = 23, model: str = 'MLP'):
        super().__init__()
        self.n_features = n_features
        self.n_targets = n_targets
        self.n_aa = A
        if self.n_targets % self.n_aa != 0:
            raise ValueError(f"n_targets ({n_targets}) must be divisible by n_aa ({self.n_aa}).")
        self.n_positions = self.n_targets // self.n_aa

        if model == 'MLP':
            self.model = MLP(self.n_features, self.n_targets)
        elif model == 'CNN':
            self.model = CNN1D(self.n_features, self.n_targets).apply(init_weights)
        else:
            raise ValueError('model must be MLP or CNN.')

    def forward(self, x):
        logits = self.model(x).reshape(-1, self.n_aa, self.n_positions)
        return logits

def CE(logits: torch.Tensor, target_probs: torch.Tensor):
    "Cross-entropy with soft labels." 
    logp = F.log_softmax(logits, dim=1)
    ce = -(target_probs * logp).sum(dim=1)
    return ce.mean()

def KLD(logits: torch.Tensor, target_probs: torch.Tensor):
    "Average KL divergence across positions between target_probs and softmax(logits)." 
    logq = F.log_softmax(logits, dim=1)
    logp = torch.log(target_probs + 1e-8)
    kl = (target_probs * (logp - logq)).sum(dim=1)
    return kl.mean()

def JSD(logits: torch.Tensor, target_probs: torch.Tensor):
    "Average Jensen-Shannon divergence across positions between target_probs and softmax(logits)." 
    q = F.softmax(logits, dim=1)
    p = target_probs
    m = 0.5 * (p + q)
    logp = torch.log(p + 1e-8)
    logq = torch.log(q + 1e-8)
    logm = torch.log(m + 1e-8)
    kld_pm = (p * (logp - logm)).sum(dim=1)
    kld_qm = (q * (logq - logm)).sum(dim=1)
    return (0.5 * (kld_pm + kld_qm)).mean()

def train_dl(
    df: pd.DataFrame,
    feat_col,
    target_col,
    split,
    model_func,
    A: int = 23,
    n_epoch: int = 4,
    bs: int = 32,
    lr: float = 1e-2,
    loss=CE,
    save=None,
    sampler=None,
    lr_find: bool = False,
):
    "Train a deep learning model with the fastai learner stack." 
    train = df.iloc[split[0]]
    valid = df.iloc[split[1]]

    train_ds = GeneralDataset(train, feat_col, target_col, A=A)
    valid_ds = GeneralDataset(valid, feat_col, target_col, A=A)
    dls = DataLoaders.from_dsets(train_ds, valid_ds, bs=bs, num_workers=min(fc.defaults.cpus, 4))

    model = model_func()
    learn = Learner(dls.to(def_device), model.to(def_device), loss_func=loss, metrics=[KLD, JSD])

    if lr_find:
        lr_suggestion = learn.lr_find()
        plt.show()
        plt.close()
        lr = lr_suggestion.valley if hasattr(lr_suggestion, 'valley') else float(lr_suggestion)
        print('lr_find selected', lr)

    print('lr in training is', lr)
    learn.fit_one_cycle(n_epoch, lr)

    if save is not None:
        learn.save(save)

    pred, target = learn.get_preds()
    pred = F.softmax(pred, dim=1).reshape(len(valid), -1)
    target = target.reshape(len(valid), -1)

    pred = pd.DataFrame(pred.detach().cpu().numpy(), index=valid.index, columns=target_col)
    target = pd.DataFrame(target.detach().cpu().numpy(), index=valid.index, columns=target_col)
    return target, pred


def train_dl_cv(df, feat_col, target_col, splits, model_func, A: int = 23, save: str | None = None, **kwargs):
    "Cross-validation training loop for deep learning models." 
    oof_frames = []
    for fold, split in enumerate(splits):
        print(f'------fold{fold}------')
        fname = f'{save}_fold{fold}' if save is not None else None
        _, pred = train_dl(df, feat_col, target_col, split, model_func, A=A, save=fname, **kwargs)
        pred['nfold'] = fold
        oof_frames.append(pred)
    return pd.concat(oof_frames).sort_index()
