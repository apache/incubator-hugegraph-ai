import sys
import os

module_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(f"module_path = {module_path}")
if module_path not in sys.path:
    sys.path.append(module_path)
print(sys.path)

import time
import argparse
import dgl
import dgl.nn as dglnn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from models import SAGE
import copy
from pyhugegraph.client import PyHugeClient
import data_process

def main(args):
    client = PyHugeClient(
        "127.0.0.1",
        "8080",
        user="admin",
        pwd="admin",
        graph="hugegraph_diy",
        # graphspace=None,
    )

    dgl_g, features, labels = data_process.hugegraph2dgl(client=client)

    train_percent = 0.7
    val_percent = 0.15
    train_mask, val_mask, test_mask = data_process.split_dataset(
        dgl_g=dgl_g, train_percent=train_percent, val_percent=val_percent
    )

    print(dgl_g)
    print(features.shape)
    print(labels.shape)
    num_ones = np.count_nonzero(train_mask == 1)
    print(num_ones / dgl_g.num_nodes())

    device = (
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )

    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    in_feats = features.shape[1]
    n_classes = args.num_classes
    n_edges = dgl_g.num_edges()
    
    print(
        """----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d"""
        % (
            n_edges,
            n_classes,
            train_mask.int().sum().item(),
            val_mask.int().sum().item(),
            test_mask.int().sum().item(),
        )
    )

    # add self loop
    dgl_g = dgl.remove_self_loop(dgl_g)
    dgl_g = dgl.add_self_loop(dgl_g)

    # dgl_g = dgl_g.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    model = SAGE(features.shape[1], 256, args.num_classes).to(device)
    best_model = copy.deepcopy(model)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"opt={opt}")
    num_partitions = 100
    sampler = dgl.dataloading.ClusterGCNSampler(
        dgl_g,
        num_partitions,
        # prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"],
    )
    print(f"sampler = {sampler}")

    # DataLoader for generic dataloading with a graph, a set of indices (any indices, like
    # partition IDs here), and a graph sampler.
    dataloader = dgl.dataloading.DataLoader(
        dgl_g,
        torch.arange(num_partitions).to(device),
        sampler,
        device=device,
        batch_size=100,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=True,
    )

    acc = 0
    no_improvement = 0
    mean = 0
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        for it, sg in enumerate(dataloader):
            # print(f"step {it}, sg = {sg}")
            sg_features = features[sg.ndata["_ID"]]
            sg_labels = labels[sg.ndata["_ID"]]
            sg_train_msak = train_mask[sg.ndata["_ID"]].bool()
            logits = model(sg, sg_features)
            train_loss = loss_fn(logits[sg_train_msak], sg_labels[sg_train_msak])
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            if it % 5 == 0:
                train_acc = MF.accuracy(
                    logits[sg_train_msak],
                    sg_labels[sg_train_msak],
                    task="multiclass",
                    num_classes=args.num_classes,
                )

                mem = torch.cuda.max_memory_allocated() / 1000000
                print(
                    "Train Loss {:.4f}| Train Acc {:.4f}| GPU Mem {:.4f} MB".format(
                        train_loss.item(), train_acc.item(), mem
                    )
                )

        model.eval()
        with torch.no_grad():
            val_logits = []
            val_labels = []
            for it, sg in enumerate(dataloader):
                sg_features = features[sg.ndata["_ID"]]
                sg_labels = labels[sg.ndata["_ID"]]
                sg_val_mask = val_mask[sg.ndata["_ID"]].bool()
                sg_test_mask = test_mask[sg.ndata["_ID"]].bool()
                logits = model(sg, sg_features)
                val_logits.append(logits[sg_val_mask])
                val_labels.append(sg_labels[sg_val_mask])
            val_logits = torch.cat(val_logits, 0)
            val_labels = torch.cat(val_labels, 0)
            valid_acc = MF.accuracy(
                val_logits,
                val_labels,
                task="multiclass",
                num_classes=args.num_classes,
            )

        if valid_acc < acc:
            no_improvement += 1
            if no_improvement == args.early_stopping:
                print("Early stop.")
                break
        else:
            no_improvement = 0
            acc = valid_acc
            best_model = copy.deepcopy(model)

        if epoch >= 3:
            mean = (mean * (epoch - 3) + (time.time() - t0)) / (epoch - 2)
            print(
                "Time(s) {:.4f} | Val Acc {:.4f} | Best Acc {:.4f}".format(
                    mean,
                    valid_acc.item(),
                    acc,
                )
            )

    best_model.eval()
    with torch.no_grad():
        test_logits = []
        test_labels = []
        for it, sg in enumerate(dataloader):
            sg_features = features[sg.ndata["_ID"]]
            sg_labels = labels[sg.ndata["_ID"]]
            sg_test_mask = test_mask[sg.ndata["_ID"]].bool()
            logits = model(sg, sg_features)
            test_logits.append(logits[sg_test_mask])
            test_labels.append(sg_labels[sg_test_mask])
        test_logits = torch.cat(test_logits, 0)
        test_labels = torch.cat(test_labels, 0)
    test_acc = MF.accuracy(
        test_logits,
        test_labels,
        task="multiclass",
        num_classes=args.num_classes,
    )
    print("Test Acc {:.4f}".format(test_acc.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APPNP")
    # register_data_args(parser)
    parser.add_argument(
        "--num-classes", type=int, default=7, help="the number of classes"
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=100,
        help="Patient epochs to wait before early stopping.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="default: GPU 0")
    parser.add_argument(
        "--n-epochs", type=int, default=1000, help="number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight for L2 loss"
    )

    args = parser.parse_args()
    print(args)

    main(args)
