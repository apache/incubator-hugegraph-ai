import gradio as gr
import random

# 返回随机值的函数
def random_value():
    return random.randint(1, 10)

# 返回所有 Textbox 的新随机值的函数
def refresh_values():
    return random_value(), random_value()

# 启动界面
with gr.Blocks() as demo:
    info_extract_template1 = gr.Textbox(
        label="Graph extract head 1", lines=15, show_copy_button=True
    )
    info_extract_template2 = gr.Textbox(
        label="Graph extract head 2", lines=15, show_copy_button=True
    )

    # 使用 load 事件在页面加载时更新 Textbox 的值
    demo.load(fn=refresh_values, outputs=[info_extract_template1, info_extract_template2])

# 启动 Gradio 界面
demo.launch()
