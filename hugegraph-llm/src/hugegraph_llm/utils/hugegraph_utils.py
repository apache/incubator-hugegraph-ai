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
import os
import json
import shutil
from datetime import datetime

from hugegraph_llm.config import huge_settings, resource_path
from hugegraph_llm.utils.log import log
from pyhugegraph.client import PyHugeClient

MAX_BACKUP_DIRS = 7
MAX_VERTICES = 100000
MAX_EDGES = 200000
BACKUP_DIR = str(os.path.join(resource_path, huge_settings.graph_name, "backup"))

def run_gremlin_query(query, fmt=True):
    res = get_hg_client().gremlin().exec(query)
    return json.dumps(res, indent=4, ensure_ascii=False) if fmt else res


def get_hg_client():
    return PyHugeClient(
        huge_settings.graph_ip,
        huge_settings.graph_port,
        huge_settings.graph_name,
        huge_settings.graph_user,
        huge_settings.graph_pwd,
        huge_settings.graph_space,
    )


def init_hg_test_data():
    client = get_hg_client()
    client.graphs().clear_graph_all_data()
    schema = client.schema()
    schema.propertyKey("name").asText().ifNotExist().create()
    schema.propertyKey("birthDate").asText().ifNotExist().create()
    schema.vertexLabel("Person").properties("name", "birthDate").useCustomizeStringId().ifNotExist().create()
    schema.vertexLabel("Movie").properties("name").useCustomizeStringId().ifNotExist().create()
    schema.edgeLabel("ActedIn").sourceLabel("Person").targetLabel("Movie").ifNotExist().create()

    schema.indexLabel("PersonByName").onV("Person").by("name").secondary().ifNotExist().create()
    schema.indexLabel("MovieByName").onV("Movie").by("name").secondary().ifNotExist().create()

    graph = client.graph()
    graph.addVertex("Person", {"name": "Al Pacino", "birthDate": "1940-04-25"}, id="Al Pacino")
    graph.addVertex(
        "Person",
        {"name": "Robert De Niro", "birthDate": "1943-08-17"},
        id="Robert De Niro",
    )
    graph.addVertex("Movie", {"name": "The Godfather"}, id="The Godfather")
    graph.addVertex("Movie", {"name": "The Godfather Part II"}, id="The Godfather Part II")
    graph.addVertex(
        "Movie",
        {"name": "The Godfather Coda The Death of Michael Corleone"},
        id="The Godfather Coda The Death of Michael Corleone",
    )

    graph.addEdge("ActedIn", "Al Pacino", "The Godfather", {})
    graph.addEdge("ActedIn", "Al Pacino", "The Godfather Part II", {})
    graph.addEdge("ActedIn", "Al Pacino", "The Godfather Coda The Death of Michael Corleone", {})
    graph.addEdge("ActedIn", "Robert De Niro", "The Godfather Part II", {})
    schema.getSchema()
    graph.close()
    return {
        "vertex": ["Person", "Movie"],
        "edge": ["ActedIn"],
        "property": ["name", "birthDate"],
        "index": ["PersonByName", "MovieByName"],
    }


def clean_hg_data():
    client = get_hg_client()
    client.graphs().clear_graph_all_data()

def create_dir_safely(path):
    if not os.path.exists(path):
        os.makedirs(path)

def backup_data():
    try:
        client = get_hg_client()

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
        del_info = manage_backup_retention()
        return f"Backup completed successfully in {backup_subdir} \n{del_info}"
    except Exception as e:  #pylint: disable=W0718
        log.critical("Backup failed: %s", e, exc_info=True)
        return f"Backup failed: {str(e)}"


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
            return f"Deleted old backup: {old_backup}"
    except Exception as e:
        log.error("Failed to manage backup retention: %s", e, exc_info=True)
        return f"manage backup retention failed: {str(e)}"
