import gradio as gr

def update_components(input_text):
    # 使用 gr.update() 动态更新组件属性
    return gr.update(value=input_text), gr.update(visible=True)

with gr.Blocks() as demo:
    text_input = gr.Textbox(label="输入文本")
    output_textbox = gr.Textbox(label="输出文本", interactive=False)
    submit_button = gr.Button("提交", visible=False)

    # 当输入框内容发生变化时，更新输出框的文本，并显示按钮
    text_input.change(fn=update_components, inputs=text_input, outputs=[output_textbox, submit_button])

demo.launch()
