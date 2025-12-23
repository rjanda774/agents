from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr

#load_dotenv(override=True)
load_dotenv()

def push(text: str):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name": {"type": "string", "description": "The user's name, if they provided it"},
            "notes": {"type": "string", "description": "Any additional information about the conversation that's worth recording to give context"},
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question that couldn't be answered"},
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

# Responses API tool schema
tools = [
    {"type": "function", "name": record_user_details_json["name"], "description": record_user_details_json["description"], "parameters": record_user_details_json["parameters"]},
    {"type": "function", "name": record_unknown_question_json["name"], "description": record_unknown_question_json["description"], "parameters": record_unknown_question_json["parameters"]},
]


class Me:
    def __init__(self):
        self.client = OpenAI()
        self.name = "Regina Janda"

        reader = PdfReader("me/ProfileLinkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text

        with open("me/summaryRJ.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def system_prompt(self) -> str:
        system_prompt = (
            f"You are acting as {self.name}. You are answering questions on {self.name}'s website, "
            f"particularly questions related to {self.name}'s career, background, skills and experience. "
            f"Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. "
            f"You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. "
            f"Be professional and engaging, as if talking to a potential client or future employer who came across the website. "
            f"If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, "
            f"even if it's about something trivial or unrelated to career. "
            f"If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "
        )

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt

    def _extract_text_and_toolcalls(self, resp):
        """
        Returns (assistant_text, tool_calls)

        tool_calls is a list of dicts like:
          {"name": "...", "arguments": "{...}", "call_id": "..."}
        """
        assistant_text_parts = []
        tool_calls = []

        # Responses API returns output items; messages contain content parts.
        for item in resp.output:
            if item.type != "message":
                continue
            for part in item.content:
                if part.type == "output_text":
                    assistant_text_parts.append(part.text)
                elif part.type == "tool_call":
                    tool_calls.append(
                        {"name": part.name, "arguments": part.arguments, "call_id": part.call_id}
                    )

        return ("".join(assistant_text_parts).strip(), tool_calls)

    def handle_tool_call(self, tool_calls):
        """
        Execute tool calls and return tool outputs in Responses API format.
        """
        outputs = []
        for tc in tool_calls:
            tool_name = tc["name"]
            args = json.loads(tc["arguments"] or "{}")
            print(f"Tool called: {tool_name}", flush=True)

            tool_fn = globals().get(tool_name)
            result = tool_fn(**args) if tool_fn else {}

            # Responses API expects tool outputs like this:
            outputs.append(
                {
                    "type": "tool_output",
                    "tool_call_id": tc["call_id"],
                    "output": json.dumps(result),
                }
            )
        return outputs

    def chat(self, message, history):
        # Gradio ChatInterface(type="messages") can include extra keys (e.g., "metadata")
        # in history items. The OpenAI/DeepSeek APIs generally only accept role/content,
        # so we sanitize the history before sending it.
        clean_history = []
        for m in (history or []):
            if isinstance(m, dict):
                clean_history.append(
                    {
                        "role": m.get("role", "user"),
                        "content": m.get("content", ""),
                    }
                )

        input_messages = (
            [{"role": "system", "content": self.system_prompt()}]
            + clean_history
            + [{"role": "user", "content": message}]
        )

        # 1) First call
        resp = self.client.responses.create(
            model="gpt-4o-mini",
            input=input_messages,
            tools=tools,
        )

        assistant_text, tool_calls = self._extract_text_and_toolcalls(resp)

        # 2) Handle tool calls (if any), continuing the conversation
        while tool_calls:
            tool_outputs = self.handle_tool_call(tool_calls)

            resp = self.client.responses.create(
                model="gpt-4o-mini",
                previous_response_id=resp.id,
                input=tool_outputs,
                tools=tools,
            )

            new_text, tool_calls = self._extract_text_and_toolcalls(resp)
            if new_text:
                # Some tool-using models emit text after tool results
                assistant_text = (assistant_text + "\n" + new_text).strip()

        return assistant_text



if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
