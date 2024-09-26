import sys
import os

module_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(f"module_path = {module_path}")
if module_path not in sys.path:
    sys.path.append(module_path)
print(sys.path)

import argparse
import time

import torch
import dgl
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy

from pyhugegraph.client import PyHugeClient

import data_process
from models import APPNP


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
    n_classes = args.class_num
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

    dgl_g = dgl_g.to(device)

    # create APPNP model
    model = APPNP(
        dgl_g,
        in_feats,
        args.hidden_sizes,
        n_classes,
        F.relu,
        args.in_drop,
        args.edge_drop,
        args.alpha,
        args.k,
    ).to(device)
    best_model = copy.deepcopy(model)

    loss_fn = torch.nn.CrossEntropyLoss()

    # use optimizer
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # initialize graph
    mean = 0
    acc = 0
    no_improvement = 0
    for epoch in range(args.n_epochs):

        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        train_loss = loss_fn(logits[train_mask], labels[train_mask])
        train_acc = torch.sum(
            logits[train_mask].argmax(dim=1) == labels[train_mask]
        ).item() / len(labels[train_mask])
        opt.zero_grad()
        train_loss.backward()
        opt.step()

        model.eval()

        with torch.no_grad():
            valid_loss = loss_fn(logits[val_mask], labels[val_mask])
            valid_acc = torch.sum(
                logits[val_mask].argmax(dim=1) == labels[val_mask]
            ).item() / len(labels[val_mask])

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
                "Time(s) {:.4f} | Train Acc {:.4f} | Train Loss {:.4f} | Val Acc {:.4f} | Val loss {:.4f} | Best Acc {:.4f}".format(
                    mean,
                    train_acc,
                    train_loss.item(),
                    valid_acc,
                    valid_loss.item(),
                    acc,
                )
            )

    best_model.eval()
    logits = best_model(features)
    test_acc = torch.sum(
        logits[test_mask].argmax(dim=1) == labels[test_mask]
    ).item() / len(labels[test_mask])
    print("Test Acc {:.4f}".format(test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APPNP")
    # register_data_args(parser)
    parser.add_argument(
        "--class-num", type=int, default=7, help="the number of classes"
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=100,
        help="Patient epochs to wait before early stopping.",
    )
    parser.add_argument(
        "--in-drop", type=float, default=0.5, help="input feature dropout"
    )
    parser.add_argument(
        "--edge-drop", type=float, default=0.5, help="edge propagation dropout"
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument(
        "--n-epochs", type=int, default=1000, help="number of training epochs"
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[64],
        help="hidden unit sizes for appnp",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of propagation steps")
    parser.add_argument("--alpha", type=float, default=0.1, help="Teleport Probability")
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight for L2 loss"
    )
    args = parser.parse_args()
    print(args)

    main(args)
