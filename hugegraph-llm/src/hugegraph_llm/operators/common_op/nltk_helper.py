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
from pathlib import Path
from typing import List, Optional, Dict
from urllib.error import URLError, HTTPError

import nltk
from nltk.corpus import stopwords

from hugegraph_llm.config import resource_path
from hugegraph_llm.utils.log import log
from nltk.corpus import stopwords


class NLTKHelper:
    _stopwords: Dict[str, Optional[List[str]]] = {
        "english": None,
        "chinese": None,
    }

    def stopwords(self, lang: str = "chinese") -> List[str]:
        """Get stopwords."""
        _hugegraph_source_dir = os.path.join(resource_path, "nltk_data")
        if _hugegraph_source_dir not in nltk.data.path:
            nltk.data.path.append(_hugegraph_source_dir)
        if self._stopwords.get(lang) is None:
            cache_dir = self.get_cache_dir()
            nltk_data_dir = os.environ.get("NLTK_DATA", cache_dir)

            # update nltk path for nltk so that it finds the data
            if nltk_data_dir not in nltk.data.path:
                nltk.data.path.append(nltk_data_dir)

            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                try:
                    log.info("Start download nltk package stopwords")
                    nltk.download("stopwords", download_dir=nltk_data_dir, quiet=False)
                    log.debug("NLTK package stopwords is already downloaded")
                except (URLError, HTTPError, PermissionError) as e:
                    log.warning("Can't download package stopwords as error: %s", e)
        try:
            self._stopwords[lang] = stopwords.words(lang)
        except LookupError as e:
            log.warning("NLTK stopwords for lang=%s not found: %s; using empty list", lang, e)
            self._stopwords[lang] = []

        # final check
        final_stopwords = self._stopwords[lang]
        if final_stopwords is None:
            return []

        return self._stopwords[lang]

    def check_nltk_data(self):
        _hugegraph_source_dir = os.path.join(resource_path, "nltk_data")
        if _hugegraph_source_dir not in nltk.data.path:
            nltk.data.path.append(_hugegraph_source_dir)

        cache_dir = self.get_cache_dir()
        nltk_data_dir = os.environ.get("NLTK_DATA", cache_dir)
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.append(nltk_data_dir)

        required_packages = {
            'punkt': False,
            'punkt_tab': False,
            'averaged_perceptron_tagger': False,
            "averaged_perceptron_tagger_eng": False}
        for package in required_packages:
            try:
                if package in ['punkt', 'punkt_tab']:
                    nltk.data.find(f'tokenizers/{package}')
                else:
                    nltk.data.find(f'taggers/{package}')
                required_packages[package] = True
            except LookupError:
                try:
                    log.info("Start download nltk package %s", package)
                    nltk.download(package, download_dir=nltk_data_dir, quiet=False)
                except (URLError, HTTPError, PermissionError) as e:
                    log.warning("Can't download package %s as error: %s", package, e)

        check_flag = all(required_packages.values())
        if not check_flag:
            for package in required_packages:
                if nltk.data.find(f'tokenizers/{package}') or nltk.data.find(f'taggers/{package}'):
                    required_packages[package] = True
                    log.debug("Package %s is already downloaded", package)

        check_flag = all(required_packages.values())
        return check_flag

    @staticmethod
    def get_cache_dir() -> str:
        """Locate a platform-appropriate cache directory for hugegraph-llm,
        and create it if it doesn't yet exist
        """
        # User override
        if "HG_AI_CACHE_DIR" in os.environ:
            path = Path(os.environ["HG_AI_CACHE_DIR"])

        # Linux, Unix, AIX, etc.
        elif os.name == "posix" and sys.platform != "darwin":
            path = Path("/tmp/hugegraph_llm")

        # Mac OS
        elif sys.platform == "darwin":
            path = Path(os.path.expanduser("~"), "Library/Caches/hugegraph_llm")

        # Windows (hopefully)
        else:
            local = os.environ.get("LOCALAPPDATA", None) or os.path.expanduser("~\\AppData\\Local")
            path = Path(local, "hugegraph_llm")

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        return str(path)


if __name__ == "__main__":
    NLTKHelper().stopwords()
