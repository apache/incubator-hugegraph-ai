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
from logging.handlers import TimedRotatingFileHandler
import os

# TODO: unify the log format in the project (include gradle(fastapi) frame)

# Set log format
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S %p"

# Configure log file path and maximum size
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FILE = os.path.join(LOG_DIR, "llm-server.log")

# Create a logger
log = logging.getLogger("llm_app")
log.setLevel(logging.DEBUG)

# Create a handler for writing to log file
file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight', interval=1,
                                        backupCount=7, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
# Add the handler, and we could use 'log.Info(xxx)' in other files
log.addHandler(file_handler)


# ANSI escape sequences for colors
class CustomConsoleHandler(logging.StreamHandler):
    COLORS = {
        "DEBUG": "\033[0;37m",  # White
        "INFO": "\033[0;32m",  # Green
        "WARNING": "\033[0;33m",  # Yellow
        "ERROR": "\033[0;31m",  # Red
        "CRITICAL": "\033[0;41m"  # Red background
    }

    def emit(self, record):
        try:
            msg = self.format(record)
            level = record.levelname
            color_prefix = self.COLORS.get(level, "\033[0;37m")  # Default to white
            color_suffix = "\033[0m"  # Reset to default
            stream = self.stream
            stream.write(color_prefix + msg + color_suffix + self.terminator)
            self.flush()
        except Exception:  # pylint: disable=broad-exception-caught
            self.handleError(record)


# Also output logs to the console, we could add a StreamHandler here (Optional)
custom_handler = CustomConsoleHandler()  # console_handler = logging.StreamHandler()
custom_handler.setLevel(logging.DEBUG)
custom_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
log.addHandler(custom_handler)
