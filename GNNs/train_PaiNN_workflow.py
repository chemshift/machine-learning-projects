
import os
import sys
import shutil
#sys.path.append("../")

from ocpmodels.preprocessing import AtomsToGraphs
from ase import io
import ase.io
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam, Adagrad, Rprop
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from nff.data import split_train_validation_test
from nff.data import Dataset, split_train_validation_test, collate_dicts, to_tensor
from nff.train import Trainer, get_trainer, get_model, load_model, loss, hooks, metrics, evaluate

import requests
import nff.data as d
from sklearn.utils import shuffle

import torch.sparse as sp

def sparsify_offsets(offsets):
    '''
        change dense offsets to sparse offsets
    '''
    
    size = offsets.size()
    offsets = offsets.squeeze()
    indices = offsets.nonzero()
    values = offsets[indices[:, 0],indices[:, 1]]
    offsets = sp.FloatTensor(indices.t(), values, size)
    
    return offsets

data_list = []

nsets = 40

for s in range(0,nsets):
    fname = "/home/gridsan/dsema/s2ef_train_200K/data/%d.extxyz" % (s)
    frames = io.read(fname, ":")
    for crystal in frames:
        data_list.append(crystal)

a2g = AtomsToGraphs(
    max_neigh=50,
    radius=5,
    r_energy=True,
    r_forces=True,
    r_distances=False,
    r_edges=True,
    r_fixed=True,
)

data_objects = a2g.convert_all(data_list, disable_tqdm=True)

nxyz_list = []
energy_list = []
edge_list = []
offsets_list = []
force_list = []

for atoms in data_objects:
    nxyz = torch.cat((atoms.atomic_numbers[..., None], atoms.pos),dim=1)
    energy = atoms.y
    force = atoms.force
    nbr_list = atoms.edge_index.t()
    offsets = (atoms.cell_offsets).float().matmul(atoms.cell)
    
    nxyz_list.append(nxyz)
    energy_list.append(energy)
    edge_list.append(nbr_list.squeeze())
    offsets_list.append(sparsify_offsets( offsets.squeeze()) )
    #offsets_list.append(torch.cat([offsets.squeeze(), -offsets.squeeze()], dim=0))
    #offsets_list.append(offsets.squeeze())
    force_list.append(force)

print("Making NFF Dataset object.")

props = {'nxyz': nxyz_list,
         'energy': energy_list,
         'energy_grad': force_list,
         'nbr_list': edge_list,
         'offsets': offsets_list # consider pbcs
}

dataset = d.Dataset(props.copy(), units='eV')
#dataset.generate_neighbor_list(cutoff=4)

train, val, test = split_train_validation_test(dataset, val_size=0.1, test_size=0.2,
                                              seed=0)

####################################################################################################
# Begin setting up training.
OUTDIR = './sandbox_painn'

if os.path.exists(OUTDIR):
    newpath = os.path.join(os.path.dirname(OUTDIR), 'backup')
    if os.path.exists(newpath):
        shutil.rmtree(newpath)
        
    shutil.move(OUTDIR, newpath)

modelparams = {"feat_dim": 64,
              "activation": "swish",
              "n_rbf": 20,
              "cutoff": 5.0,
              "num_conv": 4,
              "output_keys": ["energy"],
              "grad_keys": ["energy_grad"],
               # whether to sum outputs from all blocks in the model
               # or just the final output block. False in the original
               # implementation
              "skip_connection": {"energy": False},
               # Whether the k parameters in the Bessel basis functions
               # are learnable. False originally
              "learnable_k": False,
               # dropout rate in the convolution layers, originally 0
               "conv_dropout": 0.04,
               # dropout rate in the readout layers, originally 0
               "readout_dropout": 0.001,
               # dictionary of means to add to each output key
               # (this is optional - if you don't supply it then
               # nothing will be added)
               "means": {"energy": train.props['energy'].mean().item()},
               # dictionary of standard devations with which to 
               # multiply each output key
               # (this is optional - if you don't supply it then
               # nothing will be multiplied)
               "stddevs": {"energy": train.props['energy'].std().item()}
              }


model = get_model(modelparams, model_type="Painn")

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)

# batch size used in the original paper
BATCH_SIZE = 450

# untrained model to test for equivariant/invariant outputs
# original_model = copy.deepcopy(model)

train_loader = DataLoader(train, batch_size=BATCH_SIZE, collate_fn=collate_dicts,
                         sampler=RandomSampler(train))

val_loader = DataLoader(val, batch_size=BATCH_SIZE, collate_fn=collate_dicts)
test_loader = DataLoader(test, batch_size=BATCH_SIZE, collate_fn=collate_dicts)

# loss trade-off used in the original paper
loss_fn = loss.build_mse_loss(loss_coef={'energy_grad': 0.90, 'energy': 0.10})

trainable_params = filter(lambda p: p.requires_grad, model.parameters())

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

print("Batch size:", BATCH_SIZE)
count_parameters(model)

# learning rate used in the original paper
optimizer = Adam(trainable_params, lr=1e-3)
# optimizer = Rprop(trainable_params, lr=1e-3)

train_metrics = [
    metrics.MeanAbsoluteError('energy'),
    metrics.MeanAbsoluteError('energy_grad')
]

max_epochs = 1000

train_hooks = [
    hooks.MaxEpochHook(max_epochs),
    hooks.CSVHook(
        OUTDIR,
        metrics=train_metrics,
    ),
    hooks.PrintingHook(
        OUTDIR,
        metrics=train_metrics,
        separator = ' | ',
        time_strf='%M:%S'
    ),
    hooks.ReduceLROnPlateauHook(
        optimizer=optimizer,
        # patience in the original paper
        patience=20,
        factor=0.5,
        min_lr=1e-5,
        window_length=1,
        stop_after_min=True
    )
]

T = Trainer(
    model_path=OUTDIR,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
    checkpoint_interval=1,
    hooks=train_hooks,
    mini_batches=4
)

T.train(device=DEVICE, n_epochs=max_epochs)

# Get best model and generate plots
results, targets, val_loss = evaluate(T.get_best_model(), 
                                      test_loader, 
                                      loss_fn, 
                                      device=DEVICE)

units = {
    'energy_grad': r'eV/$\AA$',
    'energy': 'eV'
}

fig, ax_fig = plt.subplots(1, 2, figsize=(12, 6))

for ax, key in zip(ax_fig, units.keys()):
    pred_fn = torch.cat
    targ_fn = torch.cat
    if all([len(i.shape) == 0 for i in results[key]]):
        pred_fn = torch.stack
    if all([len(i.shape) == 0 for i in targets[key]]):
        targ_fn = torch.stack
        
    pred = pred_fn(results[key], dim=0).view(-1).detach().cpu().numpy()
    targ = targ_fn(targets[key], dim=0).view(-1).detach().cpu().numpy()

    mae = abs(pred-targ).mean()
    
    ax.hexbin(pred, targ, mincnt=1)
    
    lim_min = min(np.min(pred), np.min(targ)) * 1.1
    lim_max = max(np.max(pred), np.max(targ)) * 1.1
    
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')
    
    ax.plot((lim_min, lim_max),
            (lim_min, lim_max),
            color='#000000',
            zorder=-1,
            linewidth=0.5)
    
    ax.set_title(key.upper(), fontsize=14)
    ax.set_xlabel('predicted %s (%s)' % (key, units[key]), fontsize=12)
    ax.set_ylabel('target %s (%s)' % (key, units[key]), fontsize=12)
    ax.text(0.1, 0.9, 'MAE: %.2f %s' % (mae, units[key]), 
           transform=ax.transAxes, fontsize=14)

#plt.show()
plt.savefig('test.png')

print(results['energy'].shape)
print(results['energy_grad'].shape)
