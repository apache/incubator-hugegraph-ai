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
import json
import argparse

from hugegraph_llm.config import settings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate hugegraph_llm config file')
    parser.add_argument('--file_path', type=str, default='./config.json',
                        help='The generated config file path')

    args = parser.parse_args()

    file_path = args.file_path
    if os.path.exists(file_path):
        print(f"{file_path} already exists, please delete it first!")
        sys.exit(1)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    config_dict = {}
    for k, v in settings.__dict__.items():
        config_dict[k] = v
    with open(args.file_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    print(f"Generate {file_path} successfully!")
