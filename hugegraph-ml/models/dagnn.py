import sys
import os

module_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(f"module_path = {module_path}")
if module_path not in sys.path:
    sys.path.append(module_path)
print(sys.path)

import argparse

import dgl
import numpy as np
import torch
from torch.nn import functional as F
from models import DAGNN
import copy
import time
from pyhugegraph.client import PyHugeClient

import data_process

def main(args):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
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

    dgl_g = dgl.remove_self_loop(dgl_g)
    dgl_g = dgl.add_self_loop(dgl_g)

    dgl_g = dgl_g.to(device)

    # Step 2: Create model =================================================================== #
    model = DAGNN(
        k=args.k,
        in_dim=in_feats,
        hid_dim=args.hid_dim,
        out_dim=n_classes,
        dropout=args.dropout,
    )
    model = model.to(device)
    best_model = copy.deepcopy(model)

    # Step 3: Create training components ===================================================== #
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lamb)

    # Step 4: training epochs =============================================================== #

    mean = 0
    acc = 0
    no_improvement = 0
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()

        logits = model(dgl_g, features)
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
    logits = best_model(dgl_g, features)
    test_acc = torch.sum(
        logits[test_mask].argmax(dim=1) == labels[test_mask]
    ).item() / len(labels[test_mask])
    print("Test Acc {:.4f}".format(test_acc))

    return test_acc


if __name__ == "__main__":
    """
    DAGNN Model Hyperparameters
    """
    parser = argparse.ArgumentParser(description="DAGNN")
    # data source params
    parser.add_argument(
        "--num-classes", type=int, default=7, help="the number of classes"
    )
    # cuda params
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    # training params
    parser.add_argument(
        "--n-epochs", type=int, default=1000, help="number of training epochs"
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=100,
        help="Patient epochs to wait before early stopping.",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--lamb", type=float, default=0.005, help="L2 reg.")
    # model params
    parser.add_argument(
        "--k", type=int, default=12, help="Number of propagation layers."
    )
    parser.add_argument(
        "--hid-dim", type=int, default=64, help="Hidden layer dimensionalities."
    )
    parser.add_argument("--dropout", type=float, default=0.8, help="dropout")
    args = parser.parse_args()
    print(args)

    main(args)
