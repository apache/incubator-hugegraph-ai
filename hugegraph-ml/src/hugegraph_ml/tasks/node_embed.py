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
from typing import Tuple, Dict, Any

import dgl
import torch
from torch import nn
from dgl import DGLGraph
from tqdm import trange

class NodeEmbed:
    def __init__(
            self,
            graph: DGLGraph,
            graph_info: Dict[str, Any],
            model: nn.Module):
        self.graph = graph
        self.graph_info = graph_info
        self._model = model

    def _check_graph(self):
        pass

    def train_and_embed(
            self,
            add_self_loop: bool = True,
            lr:float = 1e-3,
            weight_decay: float = 0,
            n_epochs: int = 200,
            patience: int = 0,
            gpu: int = -1
    ) -> Tuple[DGLGraph, Dict[str, Any]]:
        # Set device for training
        device = "cuda:{}".format(gpu) if gpu != -1 and torch.cuda.is_available() else "cpu"
        self._model = self._model.to(device)
        self.graph = self.graph.to(device)
        # Add self-loop if required
        if add_self_loop:
            self.graph = dgl.add_self_loop(self.graph)
        # Get node features and move to device
        feat = self.graph.ndata["feat"].to(device)
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
            # Forward pass and compute loss
            loss = self._model(self.graph, feat)
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
        embed_feat = self._model.get_embedding(self.graph, feat)
        self.graph.ndata["feat"] = embed_feat
        return self.graph, self.graph_info
