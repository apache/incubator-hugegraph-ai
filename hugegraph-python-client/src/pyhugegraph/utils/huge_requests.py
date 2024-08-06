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

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urljoin

from typing import Optional
from pyhugegraph.utils.constants import Constants
from pyhugegraph.structure.huge_context import HugeContext
from pyhugegraph.api.common import HugeModule


class HugeSession(HugeModule):
    def __init__(
        self,
        ctx: HugeContext,
        retries: int = 5,
        backoff_factor: int = 1,
        status_forcelist=(500, 502, 504),
        session: Optional[requests.Session] = None,
    ):
        """
        Initialize the HugeSession object.
        :param retries: The maximum number of retries.
        :param backoff_factor: The backoff factor, used to calculate the interval between retries.
        :param status_forcelist: A list of status codes that trigger a retry.
        :param session: An optional requests.Session instance, for testing or advanced use cases.
        """
        super().__init__(ctx)
        self._retries = retries
        self._backoff_factor = backoff_factor
        self._status_forcelist = status_forcelist
        self._auth = (ctx.username, ctx.password)
        self._headers = {"Content-Type": Constants.HEADER_CONTENT_TYPE}
        self._timeout = ctx.timeout
        self._session = session if session else requests.Session()
        self.__configure_session()

    def __configure_session(self):
        """
        Configure the retry strategy and connection adapter for the session.
        """
        retry_strategy = Retry(
            total=self._retries,
            read=self._retries,
            connect=self._retries,
            backoff_factor=self._backoff_factor,
            status_forcelist=self._status_forcelist,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._session.keep_alive = False
        # self.logger.debug(
        #     "Session configured with retries=%s and backoff_factor=%s",
        #     self.retries,
        #     self.backoff_factor,
        # )

    def resolve(self, uri):
        """
        Constructs the full URL for the given URI based on the session context and API version.

        :param uri: The URI to be appended to the base URL.
        :return: The fully resolved URL as a string.
        """
        url = f"http://{self._ctx.ip}:{self._ctx.port}/"
        if self._ctx.api_version == "v3":
            url = urljoin(
                url,
                f"graphspaces/{self._ctx.graphspace}/graphs/{self._ctx.graph_name}/",
            )
        else:
            url = urljoin(url, f"graphs/{self._ctx.graph_name}/")
        return urljoin(url, uri)

    def close(self):
        self._session.close()
        # self.logger.debug("Session closed")

    def get(self, uri, **kwargs):
        try:
            response = self._session.get(
                self.resolve(uri), auth=self._auth, headers=self._headers, **kwargs
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            # self.logger.error("HTTP Request failed: %s", e)
            raise

    def post(self, uri, **kwargs):
        try:
            response = self._session.post(
                self.resolve(uri), auth=self._auth, headers=self._headers, **kwargs
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            # self.logger.error("HTTP Request failed: %s", e)
            raise
