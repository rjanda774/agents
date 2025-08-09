import gradio as gr
from dotenv import load_dotenv, find_dotenv

env_path = find_dotenv()
print ("Env path: ", env_path)
load_dotenv(dotenv_path=env_path, override=True)
print ("------------------------------------")

from research_manager import ResearchManager

async def run(query: str):
    async for chunk in ResearchManager().run(query):
        yield chunk


with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
    gr.Markdown("# Deep Research")
    query_textbox = gr.Textbox(label="What topic would you like to research?")
    run_button = gr.Button("Run", variant="primary")
    report = gr.Markdown(label="Report")
    
    run_button.click(fn=run, inputs=query_textbox, outputs=report)
    query_textbox.submit(fn=run, inputs=query_textbox, outputs=report)

ui.launch(inbrowser=True)

