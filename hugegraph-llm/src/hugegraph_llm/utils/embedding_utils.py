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


import asyncio
from typing import Any

from tqdm import tqdm

from hugegraph_llm.models.embeddings.base import BaseEmbedding


async def get_embeddings_parallel(embedding: BaseEmbedding, vids: list[str]) -> list[Any]:
    sem = asyncio.Semaphore(10)
    batch_size = 1000

    # TODO: refactor the logic here (call async method)
    async def get_embeddings_with_semaphore(vid_list: list[str]) -> Any:
        # Executes sync embedding method in a thread pool via loop.run_in_executor, combining async programming
        # with multi-threading capabilities.
        # This pattern avoids blocking the event loop and prepares for a future fully async pipeline.
        async with sem:
            loop = asyncio.get_running_loop()
            # FIXME: [PR-238] add & use async_get_texts_embedding instead of sync method
            return await loop.run_in_executor(None, embedding.get_texts_embeddings, vid_list)

    # Split vids into batches of size batch_size
    vid_batches = [vids[i:i + batch_size] for i in range(0, len(vids), batch_size)]

    # Create tasks for each batch
    tasks = [get_embeddings_with_semaphore(batch) for batch in vid_batches]

    embeddings = []
    with tqdm(total=len(tasks)) as pbar:
        for future in asyncio.as_completed(tasks):
            batch_embeddings = await future
            embeddings.extend(batch_embeddings)  # Extend the list with batch results
            pbar.update(1)
    return embeddings
