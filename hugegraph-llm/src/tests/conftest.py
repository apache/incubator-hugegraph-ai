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
import sys
import logging
import nltk

# Get project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# Add to Python path
sys.path.insert(0, project_root)
# Add src directory to Python path
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)
# Download NLTK resources
def download_nltk_resources():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        logging.info("Downloading NLTK stopwords resource...")
        nltk.download("stopwords", quiet=True)
# Download NLTK resources before tests start
download_nltk_resources()
# Set environment variable to skip external service tests
os.environ["SKIP_EXTERNAL_SERVICES"] = "true"
# Log current Python path for debugging
logging.debug("Python path: %s", sys.path)
