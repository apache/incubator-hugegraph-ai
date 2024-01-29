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

from pyhugegraph.api.common import HugeParamsBase
from pyhugegraph.utils.exceptions import NotFoundError
from pyhugegraph.utils.huge_requests import HugeSession
from pyhugegraph.utils.util import check_if_success


class MetricsManager(HugeParamsBase):
    def __init__(self, graph_instance):
        super().__init__(graph_instance)
        self.session = self.set_session(HugeSession.new_session())

    def set_session(self, session):
        self.session = session
        return session

    def close(self):
        if self.session:
            self.session.close()

    def get_all_basic_metrics(self):
        url = f"{self._host}/metrics/?type=json"
        response = self.session.get(
            url,
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_gauges_metrics(self):
        url = f"{self._host}/metrics/gauges"
        response = self.session.get(
            url,
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_counters_metrics(self):
        url = f"{self._host}/metrics/counters"
        response = self.session.get(
            url,
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_histograms_metrics(self):
        url = f"{self._host}/metrics/histograms"
        response = self.session.get(
            url,
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_meters_metrics(self):
        url = f"{self._host}/metrics/meters"
        response = self.session.get(
            url,
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_timers_metrics(self):
        url = f"{self._host}/metrics/timers"
        response = self.session.get(
            url,
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_statistics_metrics(self):
        url = f"{self._host}/metrics/statistics/?type=json"
        response = self.session.get(
            url,
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_system_metrics(self):
        url = f"{self._host}/metrics/system"
        response = self.session.get(
            url,
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_backend_metrics(self):
        url = f"{self._host}/metrics/backend"
        response = self.session.get(
            url,
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}
