import os
import gradio as gr
from typing import Dict, List
from dotenv import load_dotenv

# Load .env from explicit path
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path)

APP_THEME = gr.themes.Soft()

EXAMPLES = [
    "When are the girls club volleyball tryouts?",
    "Who do I contact about a jersey issue?",
    "What are the practice schedule details?",
    "Are there beginner volleyball classes available?",
    "What is the policy for dual sport athletes?",
    "Where is the No Panic gym located?",
    "What summer camps are available for high school players?",
    "What happens if my player gets injured mid-season?",
]


def create_demo(force_reindex: bool = False) -> gr.ChatInterface:
    from rag import VolleyballRAGChat

    print("Initializing No Panic Volleyball Assistant...")
    chat = VolleyballRAGChat()
    chat.initialize(force_reindex=force_reindex)
    print("Ready.")

    def respond(message: str, history: List[Dict[str, str]]) -> str:
        return chat.process_message(message, history)

    demo = gr.ChatInterface(
        fn=respond,
        title="No Panic Volleyball Assistant",
        description=(
            "Ask me anything about No Panic Volleyball Club — tryouts, schedules, "
            "gear, training, facilities, and more."
        ),
        examples=EXAMPLES,
        run_examples_on_click=True,
        chatbot=gr.Chatbot(
            show_label=False,
            height=560,
            layout="bubble",
        ),
        textbox=gr.Textbox(
            show_label=False,
            placeholder="Ask about tryouts, schedules, gear, training...",
            lines=1,
            max_lines=5,
            submit_btn="Send",
            stop_btn="Stop",
        ),
        fill_height=True,
        fill_width=True,
        show_progress="minimal",
    )
    demo.theme = APP_THEME
    return demo
