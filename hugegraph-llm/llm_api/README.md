# Local LLM Api

## Usage
If hugegraph-llm wants to use local LLM, you can configure it as follows.

Run the program:
```shell
python main.py \
    --model_name_or_path "Qwen/Qwen1.5-0.5B-Chat" \
    --device "cuda" \
    --port 7999
```

The LLM Section of [config.ini](../src/hugegraph_llm/config/config.ini) can be configured as follows:
```ini
[LLM]
type = local_api
llm_url = http://localhost:7999/v1/chat/completions
```
