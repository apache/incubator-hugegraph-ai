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
    """Get embeddings for texts in parallel.

    This function processes text embeddings asynchronously in parallel, using batching and semaphore
    to control concurrency, improving processing efficiency while preventing resource overuse.

    Args:
        embedding (BaseEmbedding): The embedding model instance used to compute text embeddings.
        vids (list[str]): List of texts to compute embeddings for.

    Returns:
        list[Any]: List of embedding vectors corresponding to the input texts, maintaining the same
                  order as the input vids list.

    Note:
        - Uses a semaphore to limit maximum concurrency to 10
        - Processes texts in batches of 1000
        - Displays progress using a progress bar
    """
    sem = asyncio.Semaphore(10)
    batch_size = 1000

    async def get_embeddings_async(vid_list: list[str]) -> Any:
        # Executes sync embedding method in a thread pool via loop.run_in_executor, combining async programming
        # with multi-threading capabilities.
        # This pattern avoids blocking the event loop and prepares for a future fully async pipeline.
        return await embedding.async_get_texts_embeddings(vid_list)

    # Split vids into batches of size batch_size
    vid_batches = [vids[i:i + batch_size] for i in range(0, len(vids), batch_size)]

    # Create tasks for each batch
    tasks = [get_embeddings_async(batch) for batch in vid_batches]

    embeddings = []
    with tqdm(total=len(tasks)) as pbar:
        for future in asyncio.as_completed(tasks):
            batch_embeddings = await future
            embeddings.extend(batch_embeddings)  # Extend the list with batch results
            pbar.update(1)
    return embeddings
