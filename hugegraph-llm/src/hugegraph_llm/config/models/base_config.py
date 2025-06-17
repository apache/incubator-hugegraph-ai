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

from dotenv import dotenv_values, set_key
from pydantic_settings import BaseSettings
import yaml

from hugegraph_llm.utils.log import log

dir_name = os.path.dirname
env_path = os.path.join(os.getcwd(), ".env")  # Load .env from the current working directory
config_yaml_path = os.path.join(os.getcwd(), "config.yaml")

SECRET_FIELD_NAMES = {
    "graph_pwd",
    "user_token",
    "admin_token",
    "openai_chat_api_key",
    "openai_extract_api_key",
    "openai_text2gql_api_key",
    "openai_embedding_api_key",
    "reranker_api_key",
    "qianfan_chat_api_key",
    "qianfan_chat_secret_key",
    "qianfan_chat_access_token",
    "qianfan_extract_api_key",
    "qianfan_extract_secret_key",
    "qianfan_extract_access_token",
    "qianfan_text2gql_api_key",
    "qianfan_text2gql_secret_key",
    "qianfan_text2gql_access_token",
    "qianfan_embedding_api_key",
    "qianfan_embedding_secret_key",
    "litellm_chat_api_key",
    "litellm_extract_api_key",
    "litellm_text2gql_api_key",
    "litellm_embedding_api_key",
    # Add any other specific secret field names from your models here
}


class BaseConfig(BaseSettings):
    class Config:
        env_file = env_path
        case_sensitive = False
        extra = 'ignore'  # ignore extra fields to avoid ValidationError
        env_ignore_empty = True

    def generate_env(self):
        """Handles the scenario where the configuration file does not exist,
        i.e., when the file is generated for the first time and contains no configuration information.
        """
        config_dict = self.model_dump()
        config_dict = {k.upper(): v for k, v in config_dict.items()}
        # 1. Handle .env generation
        with open(env_path, "w", encoding="utf-8") as f:
            for k, v in config_dict.items():
                # Only store secret keys in .env
                if self._is_secret_key(k):
                    if v is None:
                        f.write(f"{k}=\n")
                    else:
                        f.write(f"{k}={v}\n")
        log.info("Generate %s successfully!", env_path)

    def generate_yaml(self):
        config_dict = self.model_dump()
        config_dict = {k.upper(): v for k, v in config_dict.items()}
        current_class_name = self.__class__.__name__
        yaml_data = {}
        yaml_section_data = {}
        for k, v in config_dict.items():
            if not self._is_secret_key(k):
                yaml_section_data[k] = v
        yaml_data[current_class_name] = yaml_section_data
        with open(config_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, sort_keys=False)
        log.info("Generate %s successfully!", config_yaml_path)
        
    def update_configs(self):
        """Updates the configurations of subclasses to the .env and config.yaml files."""
        config_dict = self.model_dump()
        config_dict = {k.upper(): v for k, v in config_dict.items()}
        
        # 1. Process the .env file: update secret key information from config_dict to .env.
        env_config = dotenv_values(f"{env_path}")

        # dotenv_values make None to '', while pydantic make None to None
        # dotenv_values make integer to string, while pydantic make integer to integer
        for k, v in config_dict.items():
            if self._is_secret_key(k):
                if not (env_config[k] or v):
                    continue
                if env_config[k] == str(v):
                    continue
            log.info("Update %s: %s=%s", env_path, k, v)
            set_key(env_path, k, v if v else "", quote_mode="never")
        
        # 2. Process the config.yaml file.
        try:
            current_class_name = self.__class__.__name__
            with open(config_yaml_path, "r", encoding="utf-8") as f:
                content = f.read()
                yaml_config = yaml.safe_load(content) if content.strip() else {}
                for k, v in config_dict.items():
                    if k in env_config and not self._is_secret_key(k):
                        yaml_config[current_class_name][k] = v
            with open(config_yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(yaml_config, f)
        except yaml.YAMLError as e:
            log.error("Error parsing YAML from %s: %s", config_yaml_path, e)
        except Exception as e:
            log.error("Error loading %s: %s", config_yaml_path, e)
        

    def check_env_configs(self):
        """Synchronize configs between .env file, config.yaml file and object.

        This method performs two main operations:
        1. For .env: Updates object attributes from .env and adds missing items to .env.
        2. For config.yaml: Updates object attributes from config.yaml and adds missing non-secret items to config.yaml.
        """
        try:
            # Read the.env file and prepare object config
            env_config = dotenv_values(env_path)
            object_config_dict = {k.upper(): v for k, v in self.model_dump().items()}

            # Step 1: Update the object from .env when values differ
            self._sync_env_to_object(env_config, object_config_dict)
            # Step 2: Add missing config items to .env
            self._sync_object_to_env(env_config, object_config_dict)
        except Exception as e:
            log.error("An error occurred when checking the .env variable file: %s", str(e))
            raise

    def check_yaml_configs(self):
        object_config_dict = {k.upper(): v for k, v in self.model_dump().items()}
        try:
            current_class_name = self.__class__.__name__
            # Read the yaml.config file and prepare object config
            with open(config_yaml_path, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    yaml_file_config = {current_class_name: {}}
                else:
                    yaml_file_config = yaml.safe_load(content)
                    if not isinstance(yaml_file_config, dict):
                        log.error("Invalid YAML content in %s. Expected a dictionary.", config_yaml_path)
                        yaml_file_config = {current_class_name: {}} # Reset to a safe state
                    elif current_class_name not in yaml_file_config:
                        yaml_file_config[current_class_name] = {}

            # Step 1: Update the object from yaml.config (non-secrets)
            if yaml_file_config.get(current_class_name):
                self._sync_yaml_to_object(yaml_file_config, object_config_dict)
            
            # Step 2: Add missing non-secret config items from object to yaml.config
            # Re-fetch current_object_config as _sync_yaml_to_object might have changed it
            object_config_after_yaml_sync = {k.upper(): v for k, v in self.model_dump().items()}
            self._sync_object_to_yaml(yaml_file_config, object_config_after_yaml_sync)

        except Exception as e:
            log.error("An error occurred when checking the yaml.config variable file: %s", str(e))
            raise

    def _sync_env_to_object(self, env_config, config_dict):
        """Update object attributes from .env file values when they differ."""
        for env_key, env_value in env_config.items():
            if env_key in config_dict and self._is_secret_key(env_key):
                obj_value = config_dict[env_key]
                obj_value_str = str(obj_value) if obj_value is not None else ""

                if env_value != obj_value_str:
                    log.info("Update configuration from the file: %s=%s (Original value: %s)",
                             env_key, env_value, obj_value_str)
                    # Update the object attribute (using lowercase key)
                    setattr(self, env_key.lower(), env_value)

    def _sync_object_to_env(self, env_config, config_dict):
        """Add missing configuration items to the .env file."""
        for obj_key, obj_value in config_dict.items():
            if self._is_secret_key(obj_key) and obj_key not in env_config:
                obj_value_str = str(obj_value) if obj_value is not None else ""
                log.info("Add secret configuration item to the .env file: %s=%s",
                         obj_key, obj_value)
                # Add to .env
                set_key(env_path, obj_key, obj_value_str, quote_mode="never")

    def _sync_yaml_to_object(self, yaml_file_config, object_config):
        """Update object attributes from yaml.config file values when they differ"""
        current_class_name = self.__class__.__name__
        if current_class_name not in yaml_file_config or not isinstance(yaml_file_config[current_class_name], dict):
            return
        
        for obj_key, obj_value in object_config.items():
            if obj_key in yaml_file_config[current_class_name]:
                yaml_value = yaml_file_config[current_class_name][obj_key]
                
                if obj_value != yaml_value:
                    log.info("Update configuration from YAML file: %s=%s (Original value: %s)",
                             obj_key, yaml_value, obj_value)
                    # Update the object attribute (using lowercase key)
                    setattr(self, obj_key.lower(), yaml_value)
        
    def _sync_object_to_yaml(self, yaml_file_config, object_config):
        """Add missing configuration items to the .yaml file."""
        current_class_name = self.__class__.__name__
        
        if current_class_name not in yaml_file_config or not isinstance(yaml_file_config[current_class_name], dict):
            yaml_file_config[current_class_name] = {} # Ensure the section exists

        for obj_key, obj_value in object_config.items():
            if not self._is_secret_key(obj_key):
                if obj_key not in yaml_file_config[current_class_name] or \
                   yaml_file_config[current_class_name][obj_key] != obj_value:
                    log.info("Add/Update configuration item in YAML structure for %s: %s=%s",
                             current_class_name, obj_key, obj_value)
                    # Add to yaml.config
                    yaml_file_config[current_class_name][obj_key] = obj_value
        with open(config_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_file_config, f, sort_keys=False)
        return yaml_file_config
                
    def _is_secret_key(self, field_name: str) -> bool:
        return field_name.lower() in SECRET_FIELD_NAMES

    def __init__(self, **data):
        try:
            env_file_exists = os.path.exists(env_path)
            yaml_file_exists = os.path.exists(config_yaml_path)
            
            # Step 1: Load environment variables if file exists
            if env_file_exists:
                env_config = dotenv_values(env_path)
                for k, v in env_config.items():
                    os.environ[k] = v

            # Step 2: Init the parent class with loaded environment variables
            super().__init__(**data)
            # Step 3: Handle environment file operations after initialization
            if not env_file_exists:
                self.generate_env_and_yaml()
            else:
                # Synchronize configurations between the object and .env file
                self.check_env_configs()

            if not yaml_file_exists:
                self.generate_yaml()
            else:
                self.check_yaml_configs()
                
            log.info("The %s file was loaded. Class: %s", env_path, self.__class__.__name__)
        except Exception as e:
            log.error("An error occurred when initializing the configuration object: %s", str(e))
            raise
