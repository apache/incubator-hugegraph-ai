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

import sys
import os
from pathlib import Path

import yaml

from hugegraph_llm.utils.log import log
from hugegraph_llm.utils.anchor import get_project_root

dir_name = os.path.dirname
F_NAME = "config_prompt.yaml"
yaml_file_path = os.path.join(os.getcwd(), "src/hugegraph_llm/resources/demo", F_NAME)


class BasePromptConfig:
    graph_schema: str = ''
    extract_graph_prompt: str = ''
    default_question: str = ''
    custom_rerank_info: str = ''
    answer_prompt: str = ''
    keywords_extract_prompt: str = ''
    text2gql_graph_schema: str = ''
    gremlin_generate_prompt: str = ''
    doc_input_text: str = ''
    _language_generated: str = ''

    def ensure_yaml_file_exists(self):
        current_dir = Path.cwd().resolve()
        project_root = get_project_root()
        if current_dir == project_root:
            log.info("Current working directory is the project root, proceeding to run the app.")
        else:
            error_msg = (
                f"Current working directory is not the project root. "
                f"Please run this script from the project root directory: {project_root}\n"
                f"Current directory: {current_dir}"
            )
            log.error(error_msg)
            sys.exit(1)
        if os.path.exists(yaml_file_path):
            log.info("Loading prompt file '%s' successfully.", F_NAME)
            with open(yaml_file_path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
                # Load existing values from the YAML file into the class attributes
                for key, value in data.items():
                    setattr(self, key, value)

            # Check if the language in the .env file matches the language in the YAML file        
            env_lang = self.llm_settings.language.lower() if hasattr(self, 'llm_settings') and self.llm_settings.language else 'en'
            yaml_lang = data.get('_language_generated', 'en').lower()
            
            if env_lang.strip() != yaml_lang.strip():
                log.warning(
                    f"Prompt was changed '.env' language is '{env_lang}', "
                    f"but '{F_NAME}' was generated for '{yaml_lang}'. "
                    f"Regenerating the prompt file..."
                )
                if self.llm_settings.language.lower() == "cn":
                    self.answer_prompt = self.answer_prompt_CN
                    self.extract_graph_prompt = self.extract_graph_prompt_CN
                    self.gremlin_generate_prompt = self.gremlin_generate_prompt_CN
                    self.keywords_extract_prompt = self.keywords_extract_prompt_CN
                    self.doc_input_text = self.doc_input_text_CN
                else:
                    self.answer_prompt = self.answer_prompt_EN 
                    self.extract_graph_prompt = self.extract_graph_prompt_EN
                    self.gremlin_generate_prompt = self.gremlin_generate_prompt_EN
                    self.keywords_extract_prompt = self.keywords_extract_prompt_EN
                    self.doc_input_text = self.doc_input_text_EN
        else:
            self.generate_yaml_file()
            log.info("Prompt file '%s' doesn't exist, create it.", yaml_file_path)

    def save_to_yaml(self):
        indented_schema = "\n".join([f"  {line}" for line in self.graph_schema.splitlines()])
        indented_text2gql_schema = "\n".join([f"  {line}" for line in self.text2gql_graph_schema.splitlines()])
        indented_question = "\n".join([f"    {line}" for line in self.default_question.splitlines()])
        indented_custom_related_information = (
            "\n".join([f"    {line}" for line in self.custom_rerank_info.splitlines()])
        )
        
        indented_gremlin_prompt = "\n".join([f"  {line}" for line in self.gremlin_generate_prompt.splitlines()])
        indented_example_prompt = "\n".join([f"    {line}" for line in self.extract_graph_prompt.splitlines()])
        indented_default_answer_template = "\n".join([f"    {line}" for line in self.answer_prompt.splitlines()])
        indented_keywords_extract_template = (
            "\n".join([f"    {line}" for line in self.keywords_extract_prompt.splitlines()])
        )
        indented_doc_input_text = "\n".join([f"  {line}" for line in self.doc_input_text.splitlines()])

        indented_language_generated = "\n".join([f"  {line}" for line in self.llm_settings.language.splitlines()])

        # This can be extended to add storage fields according to the data needs to be stored
        yaml_content = f"""graph_schema: |
{indented_schema}

text2gql_graph_schema: |
{indented_text2gql_schema}

extract_graph_prompt: |
{indented_example_prompt}

default_question: |
{indented_question}

custom_rerank_info: |
{indented_custom_related_information}

answer_prompt: |
{indented_default_answer_template}

keywords_extract_prompt: |
{indented_keywords_extract_template}

gremlin_generate_prompt: |
{indented_gremlin_prompt}

doc_input_text: |
{indented_doc_input_text}

_language_generated: |
{indented_language_generated}
"""
        with open(yaml_file_path, "w", encoding="utf-8") as file:
            file.write(yaml_content)

    def generate_yaml_file(self):
        if os.path.exists(yaml_file_path):
            log.info("%s already exists, do you want to override with the default configuration? (y/n)", yaml_file_path)
            update = input()
            if update.lower() != "y":
                return

        if self.llm_settings.language.lower() == "cn":
            self.answer_prompt = self.answer_prompt_CN
            self.extract_graph_prompt = self.extract_graph_prompt_CN
            self.gremlin_generate_prompt = self.gremlin_generate_prompt_CN
            self.keywords_extract_prompt = self.keywords_extract_prompt_CN
            self.doc_input_text = self.doc_input_text_CN
        else:
            self.answer_prompt = self.answer_prompt_EN 
            self.extract_graph_prompt = self.extract_graph_prompt_EN
            self.gremlin_generate_prompt = self.gremlin_generate_prompt_EN
            self.keywords_extract_prompt = self.keywords_extract_prompt_EN
            self.doc_input_text = self.doc_input_text_EN
        self.save_to_yaml()
        log.info("Prompt file '%s' has been generated with default values.", yaml_file_path)

    def update_yaml_file(self):
        self.save_to_yaml()
        log.info("Prompt file '%s' updated successfully.", F_NAME)
