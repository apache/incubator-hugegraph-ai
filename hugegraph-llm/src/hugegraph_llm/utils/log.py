#  Licensed to the Apache Software Foundation (ASF) under one or more
#  contributor license agreements.  See the NOTICE file distributed with
#  this work for additional information regarding copyright ownership.
#  The ASF licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging
import os


from pyhugegraph.utils.log import init_logger

# Configure common settings

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "llm-server.log")
INFO = logging.INFO


# Initialize the root logger first with Rich handler
root_logger = init_logger(
    log_output=LOG_FILE,
    log_level=INFO,
    logger_name="root",
    propagate_logs=True,
    stdout_logging=True
)

# Initialize custom logger
log = init_logger(
    log_output=LOG_FILE,
    log_level=logging.DEBUG,  # Adjust level if needed
    logger_name="llm",
)

# Configure Uvicorn (FastAPI) logging
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.handlers.clear()
uvicorn_logger.handlers.extend(root_logger.handlers)
uvicorn_logger.setLevel(INFO)

# Configure Gradio logging
gradio_logger = logging.getLogger("gradio")
gradio_logger.handlers.clear()  # remove default handlers
gradio_logger.handlers.extend(root_logger.handlers)
gradio_logger.setLevel(INFO)

# Suppress `watchfiles` logging
watchfiles_logger = logging.getLogger("watchfiles")
watchfiles_logger.handlers.clear()
watchfiles_logger.handlers.extend(root_logger.handlers)
watchfiles_logger.setLevel(logging.ERROR)

