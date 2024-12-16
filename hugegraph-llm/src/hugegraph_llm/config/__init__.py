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


__all__ = ["huge_settings", "admin_settings", "llm_settings", "resource_path"]

import os

from .prompt_config import PromptConfig
from .hugegraph_config import HugeGraphConfig
from .admin_config import AdminConfig
from .llm_config import LLMConfig

prompt = PromptConfig()
prompt.ensure_yaml_file_exists()

huge_settings = HugeGraphConfig()
admin_settings = AdminConfig()
llm_settings = LLMConfig()

package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
resource_path = os.path.join(package_path, "resources")
