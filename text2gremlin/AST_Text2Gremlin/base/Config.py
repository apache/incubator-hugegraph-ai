
"""
项目配置管理模块。

负责加载和管理项目配置文件，提供各模块所需的配置参数。
"""

import json
import os
import sys


class Config:
    def __init__(self, file_path):
        self.file_path = file_path
        self.config_data = self.load_config()
        self.gen_query = self.config_data.get(
            "genQuery"
        )  # default genQuery，can be set as translate
        self.db_id = self.config_data.get("db_id")

    def load_config(self):
        with open(self.file_path, "r") as file:
            return json.load(file)

    def get_input_query_path(self):
        return self.config_data.get("input_query_path")

    def get_input_query_template_path(self):
        return self.config_data.get("input_query_template_path")

    def get_input_corpus_dir_or_file(self):
        return self.config_data.get("input_corpus_dir_or_path")

    def set_input_corpus_dir_or_file(self, dir_or_file):
        self.config_data["input_corpus_dir_or_path"] = dir_or_file

    def get_output_path(self):
        if self.gen_query:
            dir_or_file = self.config_data.get("output_query_dir_or_file")
            if os.path.isdir(dir_or_file):
                output_path = os.path.join(dir_or_file, self.db_id + ".txt")
                return output_path
            else:
                return dir_or_file
        else:
            return self.config_data.get("output_prompt_path")

    def get_output_corpus(self):
        return self.config_data.get("output_corpus_path")

    def get_schema_dict_path(self):
        return self.config_data.get("schema_dict_path")

    def get_syn_dict_path(self):
        return self.config_data.get("syn_dict_path")

    def get_db_id(self):
        return self.db_id

    def get_schema_path(self, db_id):
        schema_dict = self.config_data.get("db_schema_path")
        return schema_dict[db_id]  # todo error check

    def get_config(self, module_name):
        return self.config_data.get(module_name)
