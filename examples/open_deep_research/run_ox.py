import argparse
import os
import threading
from datetime import datetime

from dotenv import load_dotenv
from huggingface_hub import login
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer

from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    # HfApiModel,
    LiteLLMModel,
    ToolCallingAgent,
)
from smolagents.models import OpenAIServerModel

from scripts.ox_coderag import (
    OxCodeRag,
    GetRelevantCodeSnippet,
    GetFilenameOfCodeSnippet,
    ListRepositories
)


AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
]
load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question", type=str, help="for example: 'How many studio albums did Mercedes Sosa release before 2007?'"
    )
    parser.add_argument("--model-id", type=str, default="DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--chromadb-host", required=True, type=str, help="hostname or ip address where chromadb is listening")
    parser.add_argument("--chromadb-port", required=True, type=str, help="port at which chromadb is listening")
    parser.add_argument("--chromadb-collection", required=True, type=str, help="collection name in chromadb")
    parser.add_argument("--local-dir", required=True, type=str, help="local directory where index is stored")
    parser.add_argument("--llm-host", required=True, type=str, help="hostname or ip address where openai compatible llm is listening")
    parser.add_argument("--llm-port", required=True, type=str, help="port at which openai compatible llm is listening")
    parser.add_argument("--logs", required=True, type=str, help="directory where logs will be written")
    return parser.parse_args()


custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


def create_agent(chromadb_host:str, chromadb_port:str, chromadb_collection:str, local_dir:str,
                llm_host:str, llm_port:str, logs:str, model_id="DeepSeek-R1-Distill-Qwen-32B"):
    model = OpenAIServerModel(model_id,
                            api_base=f"http://{llm_host}:{llm_port}/v1",
                            api_key="notused",
                            max_completion_tokens=10000)

    text_limit = 20000
    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    WEB_TOOLS = [
        GoogleSearchTool(provider="serpapi"),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]
    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=15,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
        logs_dir=logs
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    coderag = OxCodeRag(chromadb_host, int(chromadb_port), chromadb_collection, local_dir)
    OUR_GIT_TOOLS=[GetRelevantCodeSnippet(coderag), GetFilenameOfCodeSnippet(coderag), ListRepositories(coderag)]
    our_git_agent = ToolCallingAgent(
        model=model,
        tools=OUR_GIT_TOOLS,
        max_steps = 15,
        verbosity_level=2,
        planning_interval=4,
        name="our_git_agent",
        description="""A team member that will search our confidential source code git repositories to answer your question.
    Ask him for questions regarding any of the code in our source trees.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information.""",
        provide_run_summary=True,
        logs_dir=logs
    )

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, TextInspectorTool(model, text_limit)],
        max_steps=25,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=4,
        managed_agents=[our_git_agent, text_webbrowser_agent],
        logs_dir=logs
    )

    return manager_agent


def main():
    args = parse_args()

    agent = create_agent(args.chromadb_host, args.chromadb_port, args.chromadb_collection, args.local_dir,
                            args.llm_host, int(args.llm_port), args.logs, model_id=args.model_id)

    answer = agent.run(args.question)

    print(f"Got this answer: {answer}")


if __name__ == "__main__":
    main()
