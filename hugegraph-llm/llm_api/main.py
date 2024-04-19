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


import gc
from contextlib import asynccontextmanager
from enum import Enum
from typing import Literal, List

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM, \
    Qwen2Tokenizer, GenerationConfig
from typing_extensions import TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class ChatResponse(BaseModel):
    content: str


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


class QwenChatModel:
    def __init__(self, model_name_or_path: str, device: str = "cuda", max_new_tokens: int = 512,
                 generation_config: GenerationConfig = None):
        self.model: Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map=device
        )
        self.tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.generation_config = generation_config

    @torch.inference_mode()
    async def achat(self, messages: List[Message]):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=self.max_new_tokens,
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


def torch_gc() -> None:
    r"""
    Collects GPU memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@asynccontextmanager
async def lifespan(app: "FastAPI"):  # collects GPU memory
    yield
    torch_gc()


def create_app(chat_model: "QwenChatModel") -> "FastAPI":
    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/v1/chat/completions", response_model=ChatResponse, status_code=status.HTTP_200_OK)
    async def create_chat_completion(request: ChatRequest):
        if len(request.messages) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid length")

        if len(request.messages) % 2 == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Only supports u/a/u/a/u...")

        print("* ============= [input] ============= *")
        print(request.messages[-1]["content"])

        content = await chat_model.achat(
            messages=request.messages,
        )
        print("* ============= [output] ============= *")
        print(content)

        return ChatResponse(content=content)

    return app


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Local LLM Api for Hugegraph LLM.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model name or path")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--port", type=int, default=7999, help="Port of the service")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="The max number of tokens to generate")

    args = parser.parse_args()

    model_path = args.model_name_or_path
    device = args.device
    port = args.port
    max_new_tokens = args.max_new_tokens
    chat_model = QwenChatModel(model_path, device, max_new_tokens)
    app = create_app(chat_model)
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)


if __name__ == '__main__':
    main()
