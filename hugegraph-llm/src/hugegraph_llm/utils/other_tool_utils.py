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

import time
import json
import re
import gradio as gr
import yaml

from hugegraph_llm.config import PromptConfig
from hugegraph_llm.utils.log import log
from hugegraph_llm.models.llms.ollama import OllamaClient
from hugegraph_llm.models.llms.openai import OpenAIClient
from hugegraph_llm.models.llms.qianfan import QianfanClient
from hugegraph_llm.models.llms.litellm import LiteLLMClient
def judge(answers, standard_answer, review_model_name, review_max_tokens, key, base):
    try:
        review_client = OpenAIClient(
            api_key=key,
            api_base=base,
            model_name=review_model_name,
            max_tokens=int(review_max_tokens)
        )
        review_prompt = PromptConfig.review_prompt.format(standard_answer=standard_answer)
        for _, (model_name, answer) in enumerate(answers.items(), start=1):
            review_prompt += f"### {model_name}:\n{answer.strip()}\n\n"
        log.debug("Review_prompt: %s", review_prompt)
        response = review_client.generate(prompt=review_prompt)
        log.debug("orig_review_response: %s", response)
        match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
        if match:
            response = match.group(1).strip()
        reviews = json.loads(response)
        return reviews
    except Exception as e: # pylint: disable=W0718
        log.error("Review failed: %s", str(e))
        reviews = {"error": f"Review error: {str(e)}"}
        return reviews

def parse_llm_configurations(config_text: str):
    configs = []
    lines = config_text.strip().split("\n")
    for i, line in enumerate(lines, 1):
        fields = [x.strip() for x in line.split(",")]
        if not fields:
            continue
        llm_type = fields[0]
        try:
            if llm_type == "openai":
                # openai, model_name, api_key, api_base, max_tokens
                model_name, api_key, api_base, max_tokens = fields[1:5]
                configs.append({
                    "type": "openai",
                    "model_name": model_name,
                    "api_key": api_key,
                    "api_base": api_base,
                    "max_tokens": int(max_tokens),
                })
            elif llm_type == "qianfan_wenxin":
                # qianfan_wenxin, model_name, api_key, secret_key
                model_name, api_key, secret_key = fields[1:4]
                configs.append({
                    "type": "qianfan_wenxin",
                    "model_name": model_name,
                    "api_key": api_key,
                    "secret_key": secret_key,
                })
            elif llm_type == "ollama/local":
                # ollama/local, model_name, host, port, max_tokens
                model_name, host, port, max_tokens = fields[1:5]
                configs.append({
                    "type": "ollama/local",
                    "model_name": model_name,
                    "host": host,
                    "port": int(port),
                    "max_tokens": int(max_tokens),
                })
            elif llm_type == "litellm":
                # litellm, model_name, api_key, api_base, max_tokens
                model_name, api_key, api_base, max_tokens = fields[1:5]
                configs.append({
                    "type": "litellm",
                    "model_name": model_name,
                    "api_key": api_key,
                    "api_base": api_base,
                    "max_tokens": int(max_tokens),
                })
            else:
                raise ValueError(f"Unsupported llm type '{llm_type}' in line {i}")
        except Exception as e:
            raise ValueError(f"Error parsing line {i}: {line}\nDetails: {e}") from e
    return configs

def parse_llm_configurations_from_yaml(yaml_file_path: str):
    configs = []
    with open(yaml_file_path, "r", encoding="utf-8") as f:
        raw_configs = yaml.safe_load(f)
    if not isinstance(raw_configs, list):
        raise ValueError("YAML 文件内容必须是一个 LLM 配置列表。")
    for i, config in enumerate(raw_configs, 1):
        try:
            llm_type = config.get("type")
            if llm_type == "openai":
                configs.append({
                    "type": "openai",
                    "model_name": config["model_name"],
                    "api_key": config["api_key"],
                    "api_base": config["api_base"],
                    "max_tokens": int(config["max_tokens"]),
                })
            elif llm_type == "qianfan_wenxin":
                configs.append({
                    "type": "qianfan_wenxin",
                    "model_name": config["model_name"],
                    "api_key": config["api_key"],
                    "secret_key": config["secret_key"],
                })
            elif llm_type == "ollama/local":
                configs.append({
                    "type": "ollama/local",
                    "model_name": config["model_name"],
                    "host": config["host"],
                    "port": int(config["port"]),
                    "max_tokens": int(config["max_tokens"]),
                })
            elif llm_type == "litellm":
                configs.append({
                    "type": "litellm",
                    "model_name": config["model_name"],
                    "api_key": config["api_key"],
                    "api_base": config["api_base"],
                    "max_tokens": int(config["max_tokens"]),
                })
            else:
                raise ValueError(f"不支持的 llm type '{llm_type}'，在配置第 {i} 项")
        except Exception as e:
            raise ValueError(f"解析配置第 {i} 项失败: {e}") from e

    return configs


def auto_test_llms(
        llm_configs,
        llm_configs_file,
        prompt,
        standard_answer,
        review_model_name,
        review_max_tokens,
        key,
        base,
        fmt=True
    ):
    configs = None
    if llm_configs_file and llm_configs:
        raise gr.Error("Please only choose one between file and text.")
    if llm_configs:
        configs = parse_llm_configurations(llm_configs)
    elif llm_configs_file:
        configs = parse_llm_configurations_from_yaml(llm_configs_file)
    log.debug("LLM_configs: %s", configs)
    answers = {}
    for config in configs:
        output = None
        time_start = time.perf_counter()
        try:
            if config["type"] == "openai":
                    client = OpenAIClient(
                        api_key=config["api_key"],
                        api_base=config["api_base"],
                        model_name=config["model_name"],
                        max_tokens=config["max_tokens"],
                    )
                    output = client.generate(prompt=prompt)
            elif config["type"] == "qianfan_wenxin":
                    client = QianfanClient(
                        model_name=config["model_name"],
                        api_key=config["api_key"],
                        secret_key=config["secret_key"]
                    )
                    output = client.generate(prompt=prompt)
            elif config["type"] == "ollama/local":
                    client = OllamaClient(
                        model_name=config["model_name"],
                        host=config["host"],
                        port=config["port"],
                    )
                    output = client.generate(prompt=prompt)
            elif config["type"] == "litellm":
                    client = LiteLLMClient(
                        api_key=config["api_key"],
                        api_base=config["api_base"],
                        model_name=config["model_name"],
                        max_tokens=config["max_tokens"],
                    )
                    output = client.generate(prompt=prompt)
        except Exception as e:  # pylint: disable=broad-except
            log.error("Generate failed for %s: %s", config["model_name"], e)
            output = f"[ERROR] {e}"
        time_end = time.perf_counter()
        latency = time_end - time_start
        answers[config["model_name"]] = {
            "answer": output,
            "latency": f"{round(latency, 2)}s"
        }
    reviews = judge(
        {k: v["answer"] for k, v in answers.items()},
        standard_answer,
        review_model_name,
        review_max_tokens,
        key,
        base
    )
    log.debug("reviews: %s", reviews)
    result = {}
    reviews_dict = {item["model"]: item for item in reviews} if isinstance(reviews, list) else reviews
    for model_name, infos in answers.items():
        result[model_name] = {
            "answer": infos["answer"],
            "latency": infos["latency"],
            "review": reviews_dict.get(model_name, {})
        }
    return json.dumps(result, indent=4, ensure_ascii=False) if fmt else reviews
