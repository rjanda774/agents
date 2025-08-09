import gradio as gr
from dotenv import load_dotenv, find_dotenv
env_path = find_dotenv()
load_dotenv(env_path, override=True)

from manager_agent import manager_agent
from agents import Runner, trace, gen_trace_id, ItemHelpers


async def run(query: str):
    with trace("Autonomous manager") as t:
        notifications = []
        final_output = ""    
        # result = await Runner.run(manager_agent, query)
        # print ("Result: ", result)
        # yield result.final_output
        result = Runner.run_streamed(manager_agent, query)
        async for event in result.stream_events():
            if event.type == "raw_response_event": #and isinstance(event.data, ResponseTextDeltaEvent):
                continue            
            # When the agent updates, print that
            elif event.type == "agent_updated_stream_event":
                msg = f"Agent updated: {event.new_agent.name}"
                notifications.append(msg)
                yield "\n".join(notifications), None
            # When items are generated, print them
            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    msg = f"-- Tool was called: {event.item.raw_item.name}"
                    notifications.append(msg)
                    yield "\n".join(notifications), None
                elif event.item.type == "tool_call_output_item":
                    msg = f"-- Tool output generated"
                    notifications.append(msg)
                    ## TODO:: need a better way to extract output from tools
                    final_output = event.item.output
                    yield "\n".join(notifications), None
        yield "\n".join(notifications), f"{final_output}"

def main():
    with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
        ui.queue(default_concurrency_limit=5)
        gr.Markdown("# Deep Research")
        query_textbox = gr.Textbox(label="What topic would you like to research?", value="Latest agentic AI frameworks in 2025")
        run_button = gr.Button("Run", variant="primary")
        notifications = gr.Textbox(label="Notifications", lines=1, interactive=True)
        report = gr.Markdown(label="Report")
        run_button.click(fn=run, inputs=query_textbox, outputs=[notifications, report])
        query_textbox.submit(fn=run, inputs=query_textbox, outputs=report)
    ui.launch(inbrowser=True)

if __name__ == "__main__":
    main()
