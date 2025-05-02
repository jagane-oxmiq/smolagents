import os
from smolagents import ToolCollection, CodeAgent
from mcp import StdioServerParameters
from smolagents.models import OpenAIServerModel

server_parameters = StdioServerParameters(
        command="uvx",
        args=[
          "chroma-mcp", 
          "--client-type", 
          "http", 
          "--host", 
          "127.0.0.1", 
          "--port", 
          "8000", 
          "--custom-auth-credentials",
          "notuser",
          "--ssl",
          "false"
        ]
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
                    verbosity_level=2,
                    planning_interval=4,
                    logs_dir="/tmp")

    agent.run("List the collections in this chromadb")
