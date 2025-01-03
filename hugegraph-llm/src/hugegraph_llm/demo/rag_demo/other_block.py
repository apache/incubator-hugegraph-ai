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
import json
import os
import shutil
from contextlib import asynccontextmanager
from datetime import datetime

import gradio as gr
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI

from hugegraph_llm.config import huge_settings, resource_path
from hugegraph_llm.utils.graph_index_utils import update_vid_embedding
from hugegraph_llm.utils.hugegraph_utils import init_hg_test_data, run_gremlin_query
from hugegraph_llm.utils.log import log
from pyhugegraph.client import PyHugeClient

MAX_BACKUP_DIRS = 7
MAX_VERTICES = 100000
MAX_EDGES = 200000
BACKUP_DIR = str(os.path.join(resource_path, huge_settings.graph_name, "backup"))


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


def create_dir_safely(path):
    if not os.path.exists(path):
        os.makedirs(path)

# TODO: move the logic to a separate file
def backup_data():
    try:
        client = PyHugeClient(
            huge_settings.graph_ip,
            huge_settings.graph_port,
            huge_settings.graph_name,
            huge_settings.graph_user,
            huge_settings.graph_pwd,
            huge_settings.graph_space,
        )

        create_dir_safely(BACKUP_DIR)

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = os.path.join(BACKUP_DIR, f"{date_str}")
        create_dir_safely(backup_subdir)

        files = {
            "vertices.json": f"g.V().limit({MAX_VERTICES})",
            "edges.json": f"g.E().id().limit({MAX_EDGES})",
            "schema.json": client.schema().getSchema()
        }

        for filename, query in files.items():
            with open(os.path.join(backup_subdir, filename), "w", encoding="utf-8") as f:
                data = client.gremlin().exec(query)["data"] if "schema" not in filename else query
                json.dump(data, f)

        log.info("Backup completed successfully in %s.", backup_subdir)
        manage_backup_retention()
    except Exception as e:
        log.critical("Backup failed: %s", e, exc_info=True)
        raise Exception("Failed to execute backup") from e


def manage_backup_retention():
    try:
        backup_dirs = [
            os.path.join(BACKUP_DIR, d)
            for d in os.listdir(BACKUP_DIR)
            if os.path.isdir(os.path.join(BACKUP_DIR, d))
        ]
        backup_dirs.sort(key=os.path.getctime)

        while len(backup_dirs) > MAX_BACKUP_DIRS:
            old_backup = backup_dirs.pop(0)
            shutil.rmtree(old_backup)
            log.info("Deleted old backup: %s", old_backup)
    except Exception as e:
        log.error("Failed to manage backup retention: %s", e, exc_info=True)
        raise Exception("Failed to manage backup retention") from e


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
        trigger=CronTrigger(hour=14, minute=16),
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
