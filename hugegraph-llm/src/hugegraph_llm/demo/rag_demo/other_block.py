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
from contextlib import asynccontextmanager

import gradio as gr
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI

from hugegraph_llm.utils.graph_index_utils import update_vid_embedding
from hugegraph_llm.utils.hugegraph_utils import init_hg_test_data, run_gremlin_query, backup_data
from hugegraph_llm.utils.log import log


def create_other_block():
    gr.Markdown("""## Other Tools """)
    with gr.Row():
        inp = gr.Textbox(value="g.V().limit(10)", label="Gremlin query", show_copy_button=True, lines=8)
        out = gr.Code(label="Output", language="json", elem_classes="code-container-show")
    btn = gr.Button("Run Gremlin query")
    btn.click(fn=run_gremlin_query, inputs=[inp], outputs=out)  # pylint: disable=no-member

    gr.Markdown("---")
    with gr.Accordion("Init HugeGraph test data (🚧)", open=False):
        with gr.Row():
            inp = []
            out = gr.Textbox(label="Init Graph Demo Result", show_copy_button=True)
        btn = gr.Button("(BETA) Init HugeGraph test data (🚧)")
        btn.click(fn=init_hg_test_data, inputs=inp, outputs=out)  # pylint: disable=no-member
    with gr.Accordion("Backup Graph", open=False):
        with gr.Row():
            inp = []
            out = gr.Textbox(label="Backup Graph Result", show_copy_button=True)
        btn = gr.Button("Backup Graph")
        btn.click(fn=backup_data, inputs=inp, outputs=out)


async def timely_update_vid_embedding():
    while True:
        try:
            await asyncio.to_thread(update_vid_embedding)
            log.info("rebuild_vid_index timely executed successfully.")
        except asyncio.CancelledError as ce:
            log.info("Periodic task has been cancelled due to: %s", ce)
            break
        except Exception as e:
            log.error("Failed to execute rebuild_vid_index: %s", e, exc_info=True)
            raise Exception("Failed to execute rebuild_vid_index") from e
        await asyncio.sleep(3600)


@asynccontextmanager
async def lifespan(app: FastAPI):  # pylint: disable=W0621
    log.info("Starting background scheduler...")
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        backup_data,
        trigger=CronTrigger(hour=1, minute=0),
        id="daily_backup",
        replace_existing=True
    )
    scheduler.start()

    log.info("Starting vid embedding update task...")
    embedding_task = asyncio.create_task(timely_update_vid_embedding())
    yield

    log.info("Stopping vid embedding update task...")
    embedding_task.cancel()
    try:
        await embedding_task
    except asyncio.CancelledError:
        log.info("Vid embedding update task cancelled.")

    log.info("Shutting down background scheduler...")
    scheduler.shutdown()
