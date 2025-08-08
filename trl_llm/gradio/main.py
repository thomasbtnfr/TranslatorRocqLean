import os

import gradio as gr

from trl_llm.gradio.config import GradioConfig
from trl_llm.gradio.core import load_model, unload_model, predict


def make_gradio(config: GradioConfig) -> gr.Blocks:
    with gr.Blocks(title="TranslatorRocqLean") as demo:
        with gr.Row():
            print(config.models_folder)
            chosen_model = gr.Dropdown(
                os.listdir(config.models_folder), label="Model", info="the model you want to use", interactive=True
            )
            with gr.Column():
                load_model_button = gr.Button("Load model", variant='primary')
                unload_model_button = gr.Button("Unload model")
            model_output = gr.Textbox(label="Model status")
            load_model_button.click(lambda model: load_model(config.models_folder, model), inputs=[chosen_model], outputs=model_output)
            unload_model_button.click(unload_model, outputs=model_output)

        with gr.Accordion(label="Generation options", open=False):
            with gr.Column():
                with gr.Row():
                    max_tokens = gr.Number(value=500, label="Max Tokens", minimum=1, interactive=True)
                    temperature = gr.Number(label="Temperature (1 is greedy)", value=1, minimum=0, step=100, interactive=True)
                with gr.Row():
                    top_k = gr.Number(label="Top k", minimum=-1, step=1, value=-1, interactive=True)
                    top_p = gr.Number(label="Top p", minimum=0, step=1, value=1.0, interactive=True)
                    presence_penalty = gr.Number(label="Presence penalty", value=0.0, interactive=True)
                    frequency_penalty = gr.Number(label="Frequency penalty", value=0.0, step=0.1)

        text_input = gr.Textbox(
            lines=7,
            placeholder="User's request",
            label="Input",
            interactive=True,
            elem_id="text_input"
        )

        with gr.Row():
            with gr.Column(scale=10):
                pass
            with gr.Column(scale=1, min_width=200):
                clear_button = gr.ClearButton([text_input], "Clear", elem_id="clear_btn")
            with gr.Column(scale=1, min_width=200):
                submit_button = gr.Button("Submit", elem_id="submit_btn", variant='primary')

        text_output = gr.Textbox(
            lines=15,
            label="Output",
            interactive=False,
            show_copy_button=True,
            elem_id="textbox_output",
        )
        clear_button.add(text_output)

        submit_button.click(fn=predict, inputs=[text_input, max_tokens, temperature, presence_penalty, frequency_penalty, top_p, top_k], outputs=[text_output])

        return demo


def launch_gradio(demo: gr.Blocks, port: int):
    root_path = f'{os.environ["JUPYTERHUB_SERVICE_PREFIX"]}proxy/{port}/'
    print(f'https://jupyterhub.idris.fr{root_path}')
    demo.launch(
        root_path=root_path,
        server_port=port,
        debug=True,
        show_error=True,
        share=False,
        app_kwargs={"openapi_url": f"/{port}/info"},
    )