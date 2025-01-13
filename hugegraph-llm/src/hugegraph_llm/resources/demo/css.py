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

CSS = """
footer {
  visibility: hidden
}

.code-container-edit {
    max-height: 520px;
    overflow-y: auto; /* enable scroll */
}

.code-container-show {
    max-height: 250px;
    overflow-y: auto; /* enable scroll */
}

/* FIXME: wrap code is not work as expected now */
.wrap-code {
    white-space: pre-wrap; /* CSS3 */
    white-space: -moz-pre-wrap; /* Mozilla, since 1999 */
    white-space: -pre-wrap; /* Opera 4-6 */
    white-space: -o-pre-wrap; /* Opera 7 */
    word-wrap: break-word; /* Internet Explorer 5.5+ */
}

.output-box {
    border: 1px solid #ccc;
    border-radius: 5px;
    padding: 8px;
    box-sizing: border-box;
    min-height: 50px;
    font-family: Arial, sans-serif;
    line-height: 1.5;
    margin-top: -5px;
    width: 99.5%;
    max-height: 300px;
    overflow-y: auto;
}

.output-box-label {
    font-size: 14px;
    font-weight: bold;
    margin-bottom: -5px;
}
"""
