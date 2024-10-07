import gradio as gr

# 定义按钮点击后触发的函数
def first_function(input_text):
    return f"First function received: {input_text}"

def second_function(first_output):
    # return f"Second function received: {first_output}"
    return "李虹均 天下第一"

# 创建 Gradio 界面
with gr.Blocks() as demo:
    input_text = gr.Textbox(label="Input Text")
    output_1 = gr.Textbox(label="Output of First Function")
    output_2 = gr.Textbox(label="Output of Second Function")
    button = gr.Button("Click me")

    # 点击按钮时，首先调用 first_function，然后调用 second_function
    button.click(
        first_function,
        inputs=input_text,
        outputs=output_1
    ).then(
        second_function,
        inputs=output_1,
        outputs=output_2
    )

# 运行 Gradio 界面
demo.launch()
