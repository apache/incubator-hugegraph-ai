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

from pydantic_settings import BaseSettings
import yaml

from hugegraph_llm.utils.log import log

dir_name = os.path.dirname
yaml_path = os.path.join(os.getcwd(), "config.yaml")

class BaseConfig(BaseSettings):
    class Config:
        yaml_file = yaml_path
        case_sensitive = False
        extra = 'ignore'  # Ignore extra fields to avoid ValidationError
        env_ignore_empty = True

    def generate_yaml(self):
        # Generate a YAML file based on the current configuration
        config_dict = self.model_dump()
        config_dict = {k.upper(): v for k, v in config_dict.items()}
        current_class_name = self.__class__.__name__
        yaml_data = {}
        yaml_section_data = {}
        for k, v in config_dict.items():
            yaml_section_data[k] = v
        yaml_data[current_class_name] = yaml_section_data
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, sort_keys=False)
        log.info("Generate %s successfully!", yaml_path)
        
    def update_configs(self):
        """Update the configurations of subclasses to the config.yaml files."""
        config_dict = self.model_dump()
        config_dict = {k.upper(): v for k, v in config_dict.items()}
        
        try:
            current_class_name = self.__class__.__name__
            with open(yaml_path, "r", encoding="utf-8") as f:
                content = f.read()
                yaml_config = yaml.safe_load(content) if content.strip() else {}
                for k, v in config_dict.items():
                    if k in yaml_config[current_class_name]:
                        yaml_config[current_class_name][k] = v
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(yaml_config, f)
        except yaml.YAMLError as e:
            log.error("Error parsing YAML from %s: %s", yaml_path, e)
        except Exception as e:
            log.error("Error loading %s: %s", yaml_path, e)
        

    def check_yaml_configs(self):
        """
        Synchronize configs between config.yaml file and object.
        Updates object attributes from config.yaml and adds missing items to config.yaml.
        """
        object_config_dict = {k.upper(): v for k, v in self.model_dump().items()}
        try:
            current_class_name = self.__class__.__name__
            # Read the yaml.config file and prepare object config
            with open(yaml_path, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    yaml_file_config = {current_class_name: {}}
                else:
                    yaml_file_config = yaml.safe_load(content)
                    if not isinstance(yaml_file_config, dict):
                        log.error("Invalid YAML content in %s. Expected a dictionary.", yaml_path)
                        yaml_file_config = {current_class_name: {}} # Reset to a safe state
                    elif current_class_name not in yaml_file_config:
                        yaml_file_config[current_class_name] = {}

            # Step 1: Update the object from yaml.config
            if yaml_file_config.get(current_class_name):
                self._sync_yaml_to_object(yaml_file_config, object_config_dict)
            
            # Step 2: Add missing onfig items from object to yaml.config
            # Re-fetch object_config_after_sync as _sync_yaml_to_object might have changed it
            object_config_after_sync = {k.upper(): v for k, v in self.model_dump().items()}
            self._sync_object_to_yaml(yaml_file_config, object_config_after_sync)

        except Exception as e:
            log.error("An error occurred when checking the yaml.config variable file: %s", str(e))
            raise

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
            if obj_key not in yaml_file_config[current_class_name] or \
                yaml_file_config[current_class_name][obj_key] != obj_value:
                log.info("Add/Update configuration item in YAML structure for %s: %s=%s",
                            current_class_name, obj_key, obj_value)
                # Add to yaml.config
                yaml_file_config[current_class_name][obj_key] = obj_value
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_file_config, f, sort_keys=False)
        return yaml_file_config

    def __init__(self, **data):
        try:
            yaml_file_exists = os.path.exists(yaml_path)

            # Initialize the parent class with loaded environment variables
            super().__init__(**data)
            
            # Handle environment file operations after initialization
            if not yaml_file_exists:
                self.generate_yaml()
            else:
                # Synchronize configurations between the object and yaml file
                self.check_yaml_configs()
                
            log.info("The %s file was loaded. Class: %s", yaml_path, self.__class__.__name__)
        except Exception as e:
            log.error("An error occurred when initializing the configuration object: %s", str(e))
            raise
