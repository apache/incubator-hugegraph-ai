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
from logging.handlers import TimedRotatingFileHandler

# Set log format
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S %p"


# Function to configure logging path, default is "logs/output.log"
# You could import it in "__init__.py" & use it in the whole package
def init_log(log_file="logs/output.log"):
    # Ensure the log directory exists
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    # Create a logger
    log = logging.getLogger(__name__)  # pylint: disable=redefined-outer-name
    log.setLevel(logging.INFO)

    # Create a handler for writing to log file
    file_handler = TimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    log.addHandler(file_handler)

    # ANSI escape sequences for colors
    class CustomConsoleHandler(logging.StreamHandler):
        COLORS = {
            "DEBUG": "\033[0;37m",  # White
            "INFO": "\033[0;32m",  # Green
            "WARNING": "\033[0;33m",  # Yellow
            "ERROR": "\033[0;31m",  # Red
            "CRITICAL": "\033[0;41m",  # Red background
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
            except Exception as e:  # pylint: disable=broad-exception-caught
                self.handleError(record)
                log.error(  # pylint: disable=logging-fstring-interpolation
                    f"Log Print Exception: {e}"
                )

    # Also output logs to the console
    custom_handler = CustomConsoleHandler()
    custom_handler.setLevel(logging.DEBUG)
    custom_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    log.addHandler(custom_handler)

    return log


# Default logger configuration
log = init_log()
