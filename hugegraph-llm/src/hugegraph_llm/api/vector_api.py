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


from datetime import date

from fastapi import status, APIRouter, HTTPException

from hugegraph_llm.utils.log import log

API_CALL_TRACKER = {}


# pylint: disable=too-many-statements
def vector_http_api(router: APIRouter, update_embedding_func):
    @router.post("/vector/embedding", status_code=status.HTTP_200_OK)
    def update_embedding_api(daily_limit: int = 2):
        """
        Updates the vector embedding.
        This endpoint is rate-limited. By default, it allows 2 calls per day. (Note: Not Thread-Safe!)
        The rate limit is tracked per day and resets at midnight.
        """
        today = date.today()
        for call_date in list(API_CALL_TRACKER.keys()):
            if call_date != today:
                del API_CALL_TRACKER[call_date]
        call_count = API_CALL_TRACKER.get(today, 0)
        if call_count >= daily_limit:
            log.error("Rate limit exceeded for update_vid_embedding. Maximum %d calls per day.", daily_limit)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"API call limit of {daily_limit} per day exceeded. Please try again tomorrow."
            )
        API_CALL_TRACKER[today] = call_count + 1
        result = update_embedding_func()
        return result
