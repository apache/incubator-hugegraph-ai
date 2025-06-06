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

def get_model_filename(base_name: str, model_name: str = None) -> str:
    """Generate filename based on model name."""
    if not model_name or model_name.strip() == "":
        return base_name
    # Sanitize model_name to prevent path traversal or invalid filename chars
    safe_model_name = model_name.replace("/", "_").replace("\\", "_").strip()
    return f"{safe_model_name}_{base_name}"
