# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import copy
from typing import Dict, Any

import torch
from torch import nn
from dgl import DGLGraph
from tqdm import trange

class NodeClassify:
    def __init__(
            self,
            graph: DGLGraph,
            graph_info: Dict[str, Any],
            model: nn.Module
    ):
        self.graph = graph
        self.graph_info = graph_info
        self._model = model
        self._device = ""
        self.is_trained = False

    def _check_graph(self):
        pass

    def train(
            self,
            lr: float = 1e-3,
            weight_decay: float = 0,
            n_epochs: int = 200,
            patience: int = 0,
            gpu: int = -1
    ):
        # Set device for training
        self._device = "cuda:{}".format(gpu) if gpu != -1 and torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self.graph = self.graph.to(self._device)
        # Get node features, labels, masks and move to device
        feat = self.graph.ndata['feat'].to(self._device)
        labels = self.graph.ndata['label'].to(self._device)
        train_mask = self.graph.ndata['train_mask'].to(self._device)
        val_mask = self.graph.ndata['val_mask'].to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, weight_decay=weight_decay)
        # Variables for early stopping
        best_loss = float('inf')
        best_model = None
        epochs_no_improve = 0

        # Training model
        epochs = trange(n_epochs)
        for epoch in epochs:
            self._model.train()
            optimizer.zero_grad()
            # Forward pass, get logits, compute loss
            logits = self._model(self.graph, feat)
            loss = self._model.loss(logits[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()
            # Log
            epochs.set_description("epoch {} | train loss {:.4f}".format(epoch, loss.item()))
            # Check for improvement
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model = copy.deepcopy(self._model)  # Save the best model
                epochs_no_improve = 0  # Reset counter
            else:
                epochs_no_improve += 1

            if 0 < patience <= epochs_no_improve:
                break
            torch.cuda.empty_cache()
        # Restore the best model after training
        if best_model is not None:
            self._model = best_model
        self.is_trained = True

    def evaluate(self):
        self._model.eval()
        feat = self.graph.ndata['feat'].to(self._device)
        labels = self.graph.ndata['label'].to(self._device)
        test_mask = self.graph.ndata['test_mask'].to(self._device)
        with torch.no_grad():
            logits = self._model(self.graph, feat)
            _, predicted = torch.max(logits[test_mask], dim=1)
            correct = (predicted == labels[test_mask]).sum().item()
            acc = correct / test_mask.sum().item()
        return acc
