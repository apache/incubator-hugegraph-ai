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
from logging.handlers import RotatingFileHandler
from pyhugegraph.utils import log

# Configure log directory and file
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "llm-server.log")

# Common log format
LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s %(message)s"
DATE_FORMAT = "%m/%d/%y %H:%M:%S"

# Initialize custom logger
log = log.init_logger(
    log_output=LOG_FILE,
    log_level=logging.DEBUG,  # Adjust level if needed
    logger_name="rag",
    max_log_size=20 * 1024 * 1024
)

# Create a common file handler for all logs
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=20 * 1024 * 1024, backupCount=5)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

# Configure Uvicorn (FastAPI) logging
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.setLevel(logging.INFO)
uvicorn_logger.addHandler(file_handler)

# Configure Gradio logging
gradio_logger = logging.getLogger("gradio")
gradio_logger.setLevel(logging.INFO)
gradio_logger.addHandler(file_handler)


# Suppress `watchfiles` logging
watchfiles_logger = logging.getLogger("watchfiles")
watchfiles_logger.setLevel(logging.ERROR)  # Only log errors

# Attach file handler to root logger to catch all logs
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
