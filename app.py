#app.py (HF entry point)

from foundations.app import Me
import gradio as gr

me = Me()

gr.ChatInterface(me.chat, type="messages").launch()
