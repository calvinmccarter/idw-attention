from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

METHODS = ("fc-relu", "dot", "cosine", "neg-dist", "exp-neg-dist", "inv-dist", "neglog-dist")

class SimpleTwoLayerNet(nn.Module):
    def __init__(self, method, input_shape, n_classes, n_protos, eps=1e-3, power=2):
        super(SimpleTwoLayerNet, self).__init__()
        assert method in METHODS
        self.method = method
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_protos = n_protos
        self.eps = eps
        self.power = power
        self.keys = nn.Parameter(torch.zeros([n_protos] + list(input_shape)))
        self.values = nn.Parameter(torch.zeros(n_protos, n_classes))

    def forward(self, inputs):
        x = torch.flatten(inputs, start_dim=1)
        keys = torch.flatten(self.keys, start_dim=1)
        values = self.values

        if self.method == "neglog-dist":
            dist = self.eps + torch.cdist(x, keys).pow(self.power)
            attn = 1 / dist
            attn = attn / attn.sum(dim=1, keepdim=True)
            logits = attn @ values
        elif self.method == "inv-dist":
            dist = self.eps + torch.cdist(x, keys).pow(self.power)
            attn = 1 / dist
            logits = attn.softmax(axis=1) @ values
        elif self.method == "exp-neg-dist":
            dist = torch.cdist(x, keys).pow(self.power)
            attn = torch.exp(-0.5 * dist)
            logits = attn.softmax(axis=1) @ values
        elif self.method == "neg-dist":
            dist = torch.cdist(x, keys).pow(self.power)
            attn = -1 * dist
            logits = attn.softmax(axis=1) @ values
        elif self.method == "dot":
            scaling = np.sqrt(1 / self.n_protos)
            attn = scaling * x @ keys.T
            logits = attn.softmax(axis=1) @ values
        elif self.method == "fc-relu":
            logits = (x @ keys.T).relu() @ values
        return logits

    def augment(self, inputs, c, idx=None, verbose=False):
        x = torch.flatten(inputs, start_dim=1)
        keys = torch.flatten(self.keys, start_dim=1)
        values = self.values
        assert x.shape[0] == 1
        assert c < self.n_classes
        assert self.method == "neglog-dist"

        dist = self.eps + torch.cdist(x, keys).pow(self.power)
        attn = 1 / dist
        attn_norm = attn.sum(dim=1, keepdim=True)
        attn = attn / attn_norm
        logits = attn @ values
        eta = (1 + self.eps * attn_norm.sum()) * (logits.max() - logits[0,c])
        newval = eta * F.one_hot(torch.tensor([c]), self.n_classes)
        if eta > 0:
            if idx is None:
                if verbose:
                    print(f"Fixing with new key-value:{eta.item():.4f}")
                with torch.no_grad():
                    self.keys = torch.nn.Parameter(torch.concat([self.keys, x], axis=0))
                    self.values = torch.nn.Parameter(torch.concat([self.values, newval], axis=0))
            else:
                if verbose:
                    print(f"Replacing {idx} key-value:{eta.item():.4f}")
                with torch.no_grad():
                    self.keys.data[idx, :] = x.detach().clone()
                    self.values.data[idx, :] = newval


def create_nnnet(X, y, method, n_protos, eps, power):
    input_shape = X[0, :].shape
    #len(train_df.label.unique())
    n_classes = int(1 + torch.max(y))

    model = SimpleTwoLayerNet(
        method=method, input_shape=input_shape, n_classes=n_classes, 
        n_protos=n_protos, eps=eps, power=power)
    
    mean = torch.mean(X, axis=0)
    std = torch.std(X, axis=0)
    mins = torch.min(X, axis=0)[0]
    maxs = torch.max(X, axis=0)[0]
    protos_size = [n_protos] + list(mean.shape)
    keys_np = np.random.normal(
        mean, 0.1*std, size=protos_size).astype(np.float32)
    #keys_np = np.random.uniform(
    #    mins, maxs, size=protos_size).astype(np.float32)
    #    nn.init.zeros_(self.values)
    values_np = np.zeros((n_protos, n_classes)).astype(np.float32)
    with torch.no_grad():
        if method == "fc-relu":
            nn.init.kaiming_uniform_(model.keys, nonlinearity="relu")
            nn.init.kaiming_uniform_(model.values)
        elif method == "dot":
            nn.init.xavier_uniform_(model.keys)
            nn.init.xavier_uniform_(model.values)
            nn.init.zeros_(model.keys)
            nn.init.kaiming_uniform_(model.keys)
            nn.init.kaiming_uniform_(model.values)
        else:
            model.keys.data = torch.tensor(keys_np)
            model.values.data = torch.tensor(values_np)
    return model
