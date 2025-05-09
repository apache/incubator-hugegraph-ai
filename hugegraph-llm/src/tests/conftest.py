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

import nltk

# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# 添加到 Python 路径
sys.path.insert(0, project_root)

# 添加 src 目录到 Python 路径
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)


# 下载 NLTK 资源
def download_nltk_resources():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        print("下载 NLTK stopwords 资源...")
        nltk.download("stopwords", quiet=True)


# 在测试开始前下载 NLTK 资源
download_nltk_resources()

# 设置环境变量，跳过外部服务测试
os.environ["SKIP_EXTERNAL_SERVICES"] = "true"

# 打印当前 Python 路径，用于调试
print("Python path:", sys.path)
