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
import argparse

from pathlib import Path
from hugegraph_llm.config import settings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate hugegraph_llm config file')
    parser.add_argument('--dir_path', type=str, default='.',
                        help='The generated config file path')

    args = parser.parse_args()

    dir_path = args.dir_path
    env_path = Path(dir_path) / ".env"

    if os.path.exists(env_path):
        print(f"{env_path} already exists, please delete it first!")
        sys.exit(1)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    settings.generate_env(dir_path)
