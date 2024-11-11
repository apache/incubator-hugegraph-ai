import gradio as gr
from hugegraph_llm.config import settings
from hugegraph_llm.utils.log import log
import os
import asyncio
import requests
from fastapi import Request
# Generator to simulate the tail -f behavior
async def log_stream(log_path: str):
    """
    Stream the content of a log file like `tail -f`.
    This function is asynchronous.
    """
    try:
        with open(log_path, 'r') as file:
            # Move the cursor to the end of the file
            # file.seek(0, os.SEEK_END)
            while True:
                line = file.readline()
                if line:
                    yield line
                else:
                    await asyncio.sleep(0.1)  # Non-blocking sleep
    except FileNotFoundError:
        raise Exception(f"Log file not found: {log_path}")
    except Exception as e:
        raise Exception(f"An error occurred while reading the log: {str(e)}")
            
# Functions to read each log file
def read_llm_server_log():
    try:
        with open("logs/llm-server.log", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "LLM Server log file not found."

def read_output_log():
    try:
        with open("logs/output.log", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "Output log file not found."

# Functions to clear each log file
def clear_llm_server_log():
    with open("logs/llm-server.log", "w") as f:
        f.truncate(0)  # Clear contents of the file
    return "LLM Server log cleared."

def clear_output_log():
    with open("logs/output.log", "w") as f:
        f.truncate(0)  # Clear contents of the file
    return "Output log cleared."

# Function to validate password and control access to logs
def check_password(password, request:Request):
    client_ip = request.client.host if request else "Unknown IP"
    if password == settings.log_auth_key:
        # Return logs and update visibility
        llm_log, output_log = read_llm_server_log(), read_output_log()
        visible_update = gr.update(visible=True)
        hidden_update = gr.update(visible=False)

        # Log the successful access with the IP address
        log.info(f"Logs accessed successfully from IP: {client_ip}")

        return llm_log, output_log, visible_update, visible_update, visible_update, hidden_update
    else:
        # Log the failed attempt with IP address
        log.error(f"Incorrect password attempt from IP: {client_ip}")
        return "", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value="Incorrect password. Access denied.", visible=True)

def create_log_block():
    with gr.Blocks() as demo:
        gr.Markdown("## 5. Logs Info - Password Protected")
        
        # Password input
        password_input = gr.Textbox(
            label="Enter Password",
            type="password",
            placeholder="Enter password to access logs",
        )
        
        # Error message box, initially hidden
        error_message = gr.Textbox(
            label="",
            visible=False,
            interactive=False,
            elem_classes="error-message"
        )

        # Button to submit password
        submit_button = gr.Button("Submit")

        with gr.Row(visible=False) as hidden_row:
            with gr.Column():
                # LLM Server log display, refreshes every 1 second
                gr.Markdown("### LLM Server Log")
                llm_server_log_output = gr.Textbox(
                    label="LLM Server Log (llm-server.log)",
                    lines=20,
                    value=read_llm_server_log,  # Initial value using the function
                    show_copy_button=True,
                    elem_classes="log-container",
                    every=1,  # Refresh every 1 second
                    autoscroll=True  # Enable auto-scroll
                )
                # Button to clear LLM Server log, initially hidden
                clear_llm_server_button = gr.Button("Clear LLM Server Log", visible=False)

            with gr.Column():
                # Output log display, refreshes every 1 second
                gr.Markdown("### Output Log")
                output_log_output = gr.Textbox(
                    label="Output Log (output.log)",
                    lines=20,
                    value=read_output_log,  # Initial value using the function
                    show_copy_button=True,
                    elem_classes="log-container",
                    every=1,  # Refresh every 1 second
                    autoscroll=True  # Enable auto-scroll
                )
                # Button to clear Output log, initially hidden
                clear_output_button = gr.Button("Clear Output Log", visible=False)

        # Define what happens when the password is submitted
        submit_button.click(
            fn=check_password,
            inputs=[password_input],
            outputs=[llm_server_log_output, output_log_output, hidden_row, clear_llm_server_button, clear_output_button, error_message],
        )

        # Define what happens when the Clear LLM Server Log button is clicked
        clear_llm_server_button.click(
            fn=clear_llm_server_log,
            inputs=[],
            outputs=[llm_server_log_output],
        )

        # Define what happens when the Clear Output Log button is clicked
        clear_output_button.click(
            fn=clear_output_log,
            inputs=[],
            outputs=[output_log_output],
        )