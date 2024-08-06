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

from typing import Optional
from pyhugegraph.structure.huge_context import HugeContext
from pyhugegraph.api.common import HugeModule
from pyhugegraph.utils.huge_requests import HugeSession


class HugeComponent(HugeModule):
    def __init__(self, ctx: HugeContext, sess: Optional[HugeSession] = None):
        super().__init__(ctx)
        self._sess = sess if sess else HugeSession(ctx)

    def close(self):
        self._sess.close()
