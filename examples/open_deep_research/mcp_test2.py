import os
from smolagents import ToolCollection, CodeAgent
from mcp import StdioServerParameters
from smolagents.models import OpenAIServerModel

AUTHORIZED_IMPORTS = [
    "json",
]

server_parameters = StdioServerParameters(
        command="./github-mcp-server",
        args=[
            "--read-only",
            "stdio" 
        ],
        env={
            'GITHUB_PERSONAL_ACCESS_TOKEN': 'blahblahblah',
            'GITHUB_TOOLSETS': 'issues'
            }
    )

model_id = 'QwQ-32B'
llm_host = '127.0.0.1'
llm_port = '8080'

model = OpenAIServerModel(model_id,
                            api_base=f"http://{llm_host}:{llm_port}/v1",
                            api_key="notused",
                            max_completion_tokens=10000)
with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
    agent = CodeAgent(
                    model=model,
                    tools=[*tool_collection.tools], add_base_tools=True,
                    max_steps=25,
                    additional_authorized_imports=AUTHORIZED_IMPORTS,
                    verbosity_level=2,
                    planning_interval=4,
                    logs_dir="/tmp")

    agent.run("Find out the name of this github user")
