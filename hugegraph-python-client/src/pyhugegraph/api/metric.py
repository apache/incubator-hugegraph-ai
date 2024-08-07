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
from pyhugegraph.utils.util import check_if_success


class MetricsManager(HugeParamsBase):

    def get_all_basic_metrics(self):
        uri = "/metrics/?type=json"
        response = self._sess.get(uri)
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_gauges_metrics(self):
        uri = "/metrics/gauges"
        response = self._sess.get(uri)
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_counters_metrics(self):
        uri = "/metrics/counters"
        response = self._sess.get(uri)
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_histograms_metrics(self):
        uri = "metrics/histograms"
        response = self._sess.get(uri)
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_meters_metrics(self):
        uri = "/metrics/meters"
        response = self._sess.get(uri)
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_timers_metrics(self):
        uri = "/metrics/timers"
        response = self._sess.get(uri)
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_statistics_metrics(self):
        uri = '/metrics/statistics/?type=json'
        response = self._sess.get(uri)
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_system_metrics(self):
        uri = '/metrics/system'
        response = self._sess.get(uri)
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_backend_metrics(self):
        uri = '/metrics/backend'
        response = self._sess.get(uri)
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}
