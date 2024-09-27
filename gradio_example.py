import gradio as gr

# 假设这是你的数据源
class Prompt:
    extract_graph_prompt1 = "初始值1"
    extract_graph_prompt2 = "初始值2"

prompt = Prompt()

# 返回所有 Textbox 的新值的函数
def refresh_values():
    # 在这里可以更新 prompt 的值
    prompt.extract_graph_prompt1 = "新值1"
    prompt.extract_graph_prompt2 = "新值2"
    return prompt.extract_graph_prompt1, prompt.extract_graph_prompt2

# 启动界面
with gr.Blocks() as demo:
    info_extract_template1 = gr.Textbox(
        label="Graph extract head 1", lines=15, show_copy_button=True,
        value=prompt.extract_graph_prompt1
    )
    info_extract_template2 = gr.Textbox(
        label="Graph extract head 2", lines=15, show_copy_button=True,
        value=prompt.extract_graph_prompt2
    )

    refresh_button = gr.Button("刷新所有组件")

    # 设置按钮的回调，将新值输出到 Textbox
    refresh_button.click(fn=refresh_values, outputs=[info_extract_template1, info_extract_template2])

# 启动 Gradio 界面
demo.launch()
