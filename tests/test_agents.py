# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import tempfile
import unittest
import uuid
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from smolagents.agent_types import AgentImage, AgentText
from smolagents.agents import (
    AgentMaxStepsError,
    CodeAgent,
    MultiStepAgent,
    ToolCall,
    ToolCallingAgent,
    populate_template,
)
from smolagents.default_tools import DuckDuckGoSearchTool, FinalAnswerTool, PythonInterpreterTool, VisitWebpageTool
from smolagents.memory import ActionStep, PlanningStep
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallDefinition,
    HfApiModel,
    MessageRole,
    TransformersModel,
)
from smolagents.tools import Tool, tool
from smolagents.utils import BASE_BUILTIN_MODULES, AgentGenerationError


def get_new_path(suffix="") -> str:
    directory = tempfile.mkdtemp()
    return os.path.join(directory, str(uuid.uuid4()) + suffix)


class FakeToolCallModel:
    def __call__(self, messages, tools_to_call_from=None, stop_sequences=None, grammar=None):
        if len(messages) < 3:
            return ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_0",
                        type="function",
                        function=ChatMessageToolCallDefinition(
                            name="python_interpreter", arguments={"code": "2*3.6452"}
                        ),
                    )
                ],
            )
        else:
            return ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallDefinition(name="final_answer", arguments={"answer": "7.2904"}),
                    )
                ],
            )


class FakeToolCallModelImage:
    def __call__(self, messages, tools_to_call_from=None, stop_sequences=None, grammar=None):
        if len(messages) < 3:
            return ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_0",
                        type="function",
                        function=ChatMessageToolCallDefinition(
                            name="fake_image_generation_tool",
                            arguments={"prompt": "An image of a cat"},
                        ),
                    )
                ],
            )
        else:
            return ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallDefinition(name="final_answer", arguments="image.png"),
                    )
                ],
            )


class FakeToolCallModelVL:
    def __call__(self, messages, tools_to_call_from=None, stop_sequences=None, grammar=None):
        if len(messages) < 3:
            return ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_0",
                        type="function",
                        function=ChatMessageToolCallDefinition(
                            name="fake_image_understanding_tool",
                            arguments={
                                "prompt": "What is in this image?",
                                "image": "image.png",
                            },
                        ),
                    )
                ],
            )
        else:
            return ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallDefinition(name="final_answer", arguments="The image is a cat."),
                    )
                ],
            )


def fake_code_model(messages, stop_sequences=None, grammar=None) -> str:
    prompt = str(messages)
    if "special_marker" not in prompt:
        return ChatMessage(
            role="assistant",
            content="""
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
result = 2**3.6452
```<end_code>
""",
        )
    else:  # We're at step 2
        return ChatMessage(
            role="assistant",
            content="""
Thought: I can now answer the initial question
Code:
```py
final_answer(7.2904)
```<end_code>
""",
        )


def fake_code_model_error(messages, stop_sequences=None) -> str:
    prompt = str(messages)
    if "special_marker" not in prompt:
        return ChatMessage(
            role="assistant",
            content="""
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
print("Flag!")
def error_function():
    raise ValueError("error")

error_function()
```<end_code>
""",
        )
    else:  # We're at step 2
        return ChatMessage(
            role="assistant",
            content="""
Thought: I faced an error in the previous step.
Code:
```py
final_answer("got an error")
```<end_code>
""",
        )


def fake_code_model_syntax_error(messages, stop_sequences=None) -> str:
    prompt = str(messages)
    if "special_marker" not in prompt:
        return ChatMessage(
            role="assistant",
            content="""
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
a = 2
b = a * 2
    print("Failing due to unexpected indent")
print("Ok, calculation done!")
```<end_code>
""",
        )
    else:  # We're at step 2
        return ChatMessage(
            role="assistant",
            content="""
Thought: I can now answer the initial question
Code:
```py
final_answer("got an error")
```<end_code>
""",
        )


def fake_code_model_import(messages, stop_sequences=None) -> str:
    return ChatMessage(
        role="assistant",
        content="""
Thought: I can answer the question
Code:
```py
import numpy as np
final_answer("got an error")
```<end_code>
""",
    )


def fake_code_functiondef(messages, stop_sequences=None) -> str:
    prompt = str(messages)
    if "special_marker" not in prompt:
        return ChatMessage(
            role="assistant",
            content="""
Thought: Let's define the function. special_marker
Code:
```py
import numpy as np

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
```<end_code>
""",
        )
    else:  # We're at step 2
        return ChatMessage(
            role="assistant",
            content="""
Thought: I can now answer the initial question
Code:
```py
x, w = [0, 1, 2, 3, 4, 5], 2
res = moving_average(x, w)
final_answer(res)
```<end_code>
""",
        )


def fake_code_model_single_step(messages, stop_sequences=None, grammar=None) -> str:
    return ChatMessage(
        role="assistant",
        content="""
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
result = python_interpreter(code="2*3.6452")
final_answer(result)
```
""",
    )


def fake_code_model_no_return(messages, stop_sequences=None, grammar=None) -> str:
    return ChatMessage(
        role="assistant",
        content="""
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
result = python_interpreter(code="2*3.6452")
print(result)
```
""",
    )


class AgentTests(unittest.TestCase):
    def test_fake_toolcalling_agent(self):
        agent = ToolCallingAgent(tools=[PythonInterpreterTool()], model=FakeToolCallModel())
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, str)
        assert "7.2904" in output
        assert agent.memory.steps[0].task == "What is 2 multiplied by 3.6452?"
        assert "7.2904" in agent.memory.steps[1].observations
        assert agent.memory.steps[2].model_output is None

    def test_toolcalling_agent_handles_image_tool_outputs(self):
        import PIL.Image

        @tool
        def fake_image_generation_tool(prompt: str) -> PIL.Image.Image:
            """Tool that generates an image.

            Args:
                prompt: The prompt
            """
            from pathlib import Path

            import PIL.Image

            return PIL.Image.open(Path("tests/data/000000039769.png"))

        agent = ToolCallingAgent(tools=[fake_image_generation_tool], model=FakeToolCallModelImage())
        output = agent.run("Make me an image.")
        assert isinstance(output, AgentImage)
        assert isinstance(agent.state["image.png"], PIL.Image.Image)

    def test_toolcalling_agent_handles_image_inputs(self):
        import PIL.Image

        image = PIL.Image.open(Path("tests/data/000000039769.png"))  # dummy input

        @tool
        def fake_image_understanding_tool(prompt: str, image: PIL.Image.Image) -> str:
            """Tool that creates a caption for an image.

            Args:
                prompt: The prompt
                image: The image
            """
            return "The image is a cat."

        agent = ToolCallingAgent(tools=[fake_image_understanding_tool], model=FakeToolCallModelVL())
        output = agent.run("Caption this image.", images=[image])
        assert output == "The image is a cat."

    def test_fake_code_agent(self):
        agent = CodeAgent(tools=[PythonInterpreterTool()], model=fake_code_model)
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, float)
        assert output == 7.2904
        assert agent.memory.steps[0].task == "What is 2 multiplied by 3.6452?"
        assert agent.memory.steps[2].tool_calls == [
            ToolCall(name="python_interpreter", arguments="final_answer(7.2904)", id="call_2")
        ]

    def test_additional_args_added_to_task(self):
        agent = CodeAgent(tools=[], model=fake_code_model)
        agent.run(
            "What is 2 multiplied by 3.6452?",
            additional_args={"instruction": "Remember this."},
        )
        assert "Remember this" in agent.task
        assert "Remember this" in str(agent.input_messages)

    def test_reset_conversations(self):
        agent = CodeAgent(tools=[PythonInterpreterTool()], model=fake_code_model)
        output = agent.run("What is 2 multiplied by 3.6452?", reset=True)
        assert output == 7.2904
        assert len(agent.memory.steps) == 3

        output = agent.run("What is 2 multiplied by 3.6452?", reset=False)
        assert output == 7.2904
        assert len(agent.memory.steps) == 5

        output = agent.run("What is 2 multiplied by 3.6452?", reset=True)
        assert output == 7.2904
        assert len(agent.memory.steps) == 3

    def test_setup_agent_with_empty_toolbox(self):
        ToolCallingAgent(model=FakeToolCallModel(), tools=[])

    def test_fails_max_steps(self):
        agent = CodeAgent(
            tools=[PythonInterpreterTool()],
            model=fake_code_model_no_return,  # use this callable because it never ends
            max_steps=5,
        )
        answer = agent.run("What is 2 multiplied by 3.6452?")
        assert len(agent.memory.steps) == 7  # Task step + 5 action steps + Final answer
        assert type(agent.memory.steps[-1].error) is AgentMaxStepsError
        assert isinstance(answer, str)

        agent = CodeAgent(
            tools=[PythonInterpreterTool()],
            model=fake_code_model_no_return,  # use this callable because it never ends
            max_steps=5,
        )
        answer = agent.run("What is 2 multiplied by 3.6452?", max_steps=3)
        assert len(agent.memory.steps) == 5  # Task step + 3 action steps + Final answer
        assert type(agent.memory.steps[-1].error) is AgentMaxStepsError
        assert isinstance(answer, str)

    def test_tool_descriptions_get_baked_in_system_prompt(self):
        tool = PythonInterpreterTool()
        tool.name = "fake_tool_name"
        tool.description = "fake_tool_description"
        agent = CodeAgent(tools=[tool], model=fake_code_model)
        agent.run("Empty task")
        assert tool.name in agent.system_prompt
        assert tool.description in agent.system_prompt

    def test_module_imports_get_baked_in_system_prompt(self):
        agent = CodeAgent(tools=[], model=fake_code_model)
        agent.run("Empty task")
        for module in BASE_BUILTIN_MODULES:
            assert module in agent.system_prompt

    def test_init_agent_with_different_toolsets(self):
        toolset_1 = []
        agent = CodeAgent(tools=toolset_1, model=fake_code_model)
        assert len(agent.tools) == 1  # when no tools are provided, only the final_answer tool is added by default

        toolset_2 = [PythonInterpreterTool(), PythonInterpreterTool()]
        with pytest.raises(ValueError) as e:
            agent = CodeAgent(tools=toolset_2, model=fake_code_model)
        assert "Each tool or managed_agent should have a unique name!" in str(e)

        with pytest.raises(ValueError) as e:
            agent.name = "python_interpreter"
            agent.description = "empty"
            CodeAgent(tools=[PythonInterpreterTool()], model=fake_code_model, managed_agents=[agent])
        assert "Each tool or managed_agent should have a unique name!" in str(e)

        # check that python_interpreter base tool does not get added to CodeAgent
        agent = CodeAgent(tools=[], model=fake_code_model, add_base_tools=True)
        assert len(agent.tools) == 3  # added final_answer tool + search + visit_webpage

        # check that python_interpreter base tool gets added to ToolCallingAgent
        agent = ToolCallingAgent(tools=[], model=fake_code_model, add_base_tools=True)
        assert len(agent.tools) == 4  # added final_answer tool + search + visit_webpage

    def test_function_persistence_across_steps(self):
        agent = CodeAgent(
            tools=[],
            model=fake_code_functiondef,
            max_steps=2,
            additional_authorized_imports=["numpy"],
        )
        res = agent.run("ok")
        assert res[0] == 0.5

    def test_init_managed_agent(self):
        agent = CodeAgent(tools=[], model=fake_code_functiondef, name="managed_agent", description="Empty")
        assert agent.name == "managed_agent"
        assert agent.description == "Empty"

    def test_agent_description_gets_correctly_inserted_in_system_prompt(self):
        managed_agent = CodeAgent(tools=[], model=fake_code_functiondef, name="managed_agent", description="Empty")
        manager_agent = CodeAgent(
            tools=[],
            model=fake_code_functiondef,
            managed_agents=[managed_agent],
        )
        assert "You can also give tasks to team members." not in managed_agent.system_prompt
        assert "{{managed_agents_descriptions}}" not in managed_agent.system_prompt
        assert "You can also give tasks to team members." in manager_agent.system_prompt

    def test_replay_shows_logs(self):
        agent = CodeAgent(
            tools=[], model=fake_code_model_import, verbosity_level=0, additional_authorized_imports=["numpy"]
        )
        agent.run("Count to 3")

        with agent.logger.console.capture() as capture:
            agent.replay()
        str_output = capture.get().replace("\n", "")
        assert "New run" in str_output
        assert "Agent output:" in str_output
        assert 'final_answer("got' in str_output
        assert "```<end_code>" in str_output

    def test_code_nontrivial_final_answer_works(self):
        def fake_code_model_final_answer(messages, stop_sequences=None, grammar=None):
            return ChatMessage(
                role="assistant",
                content="""Code:
```py
def nested_answer():
    final_answer("Correct!")

nested_answer()
```<end_code>""",
            )

        agent = CodeAgent(tools=[], model=fake_code_model_final_answer)

        output = agent.run("Count to 3")
        assert output == "Correct!"

    def test_transformers_toolcalling_agent(self):
        @tool
        def weather_api(location: str, celsius: bool = False) -> str:
            """
            Gets the weather in the next days at given location.
            Secretly this tool does not care about the location, it hates the weather everywhere.

            Args:
                location: the location
                celsius: the temperature type
            """
            return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

        model = TransformersModel(
            model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
            max_new_tokens=100,
            device_map="auto",
            do_sample=False,
        )
        agent = ToolCallingAgent(model=model, tools=[weather_api], max_steps=1, verbosity_level=10)
        agent.run("What's the weather in Paris?")
        assert agent.memory.steps[0].task == "What's the weather in Paris?"
        assert agent.memory.steps[1].tool_calls[0].name == "weather_api"
        step_memory_dict = agent.memory.get_succinct_steps()[1]
        assert step_memory_dict["model_output_message"].tool_calls[0].function.name == "weather_api"
        assert step_memory_dict["model_output_message"].raw["completion_kwargs"]["max_new_tokens"] == 100
        assert "model_input_messages" in agent.memory.get_full_steps()[1]

    def test_final_answer_checks(self):
        def check_always_fails(final_answer, agent_memory):
            assert False, "Error raised in check"

        agent = CodeAgent(model=fake_code_model, tools=[], final_answer_checks=[check_always_fails])
        agent.run("Dummy task.")
        assert "Error raised in check" in str(agent.write_memory_to_messages())

    def test_generation_errors_are_raised(self):
        def fake_model(messages, stop_sequences):
            assert False, "Generation failed"

        agent = CodeAgent(model=fake_model, tools=[])
        with pytest.raises(AgentGenerationError) as e:
            agent.run("Dummy task.")
        assert len(agent.memory.steps) == 2
        assert "Generation failed" in str(e)


class CustomFinalAnswerTool(FinalAnswerTool):
    def forward(self, answer) -> str:
        return answer + "CUSTOM"


class MockTool(Tool):
    def __init__(self, name):
        self.name = name
        self.description = "Mock tool description"
        self.inputs = {}
        self.output_type = "string"

    def forward(self):
        return "Mock tool output"


class MockAgent:
    def __init__(self, name, tools, description="Mock agent description"):
        self.name = name
        self.tools = {t.name: t for t in tools}
        self.description = description


class TestMultiStepAgent:
    def test_instantiation_disables_logging_to_terminal(self):
        fake_model = MagicMock()
        agent = MultiStepAgent(tools=[], model=fake_model)
        assert agent.logger.level == -1, "logging to terminal should be disabled for testing using a fixture"

    def test_instantiation_with_prompt_templates(self, prompt_templates):
        agent = MultiStepAgent(tools=[], model=MagicMock(), prompt_templates=prompt_templates)
        assert agent.prompt_templates == prompt_templates
        assert agent.prompt_templates["system_prompt"] == "This is a test system prompt."
        assert "managed_agent" in agent.prompt_templates
        assert agent.prompt_templates["managed_agent"]["task"] == "Task for {{name}}: {{task}}"
        assert agent.prompt_templates["managed_agent"]["report"] == "Report for {{name}}: {{final_answer}}"

    @pytest.mark.parametrize(
        "tools, expected_final_answer_tool",
        [([], FinalAnswerTool), ([CustomFinalAnswerTool()], CustomFinalAnswerTool)],
    )
    def test_instantiation_with_final_answer_tool(self, tools, expected_final_answer_tool):
        agent = MultiStepAgent(tools=tools, model=MagicMock())
        assert "final_answer" in agent.tools
        assert isinstance(agent.tools["final_answer"], expected_final_answer_tool)

    def test_logs_display_thoughts_even_if_error(self):
        def fake_json_model_no_call(messages, stop_sequences=None, tools_to_call_from=None):
            return ChatMessage(
                role="assistant",
                content="""I don't want to call tools today""",
                tool_calls=None,
                raw="""I don't want to call tools today""",
            )

        agent_toolcalling = ToolCallingAgent(model=fake_json_model_no_call, tools=[], max_steps=1, verbosity_level=10)
        with agent_toolcalling.logger.console.capture() as capture:
            agent_toolcalling.run("Dummy task")
        assert "don't" in capture.get() and "want" in capture.get()

        def fake_code_model_no_call(messages, stop_sequences=None):
            return ChatMessage(
                role="assistant",
                content="""I don't want to write an action today""",
            )

        agent_code = CodeAgent(model=fake_code_model_no_call, tools=[], max_steps=1, verbosity_level=10)
        with agent_code.logger.console.capture() as capture:
            agent_code.run("Dummy task")
        assert "don't" in capture.get() and "want" in capture.get()

    def test_step_number(self):
        fake_model = MagicMock()
        fake_model.last_input_token_count = 10
        fake_model.last_output_token_count = 20
        max_steps = 2
        agent = MultiStepAgent(tools=[], model=fake_model, max_steps=max_steps)
        assert hasattr(agent, "step_number"), "step_number attribute should be defined"
        assert agent.step_number == 0, "step_number should be initialized to 0"
        agent.run("Test task")
        assert hasattr(agent, "step_number"), "step_number attribute should be defined"
        assert agent.step_number == max_steps + 1, "step_number should be max_steps + 1 after run method is called"

    @pytest.mark.parametrize(
        "step, expected_messages_list",
        [
            (
                1,
                [
                    [{"role": MessageRole.USER, "content": [{"type": "text", "text": "INITIAL_PLAN_USER_PROMPT"}]}],
                ],
            ),
            (
                2,
                [
                    [
                        {
                            "role": MessageRole.SYSTEM,
                            "content": [{"type": "text", "text": "UPDATE_PLAN_SYSTEM_PROMPT"}],
                        },
                        {"role": MessageRole.USER, "content": [{"type": "text", "text": "UPDATE_PLAN_USER_PROMPT"}]},
                    ],
                ],
            ),
        ],
    )
    def test_planning_step(self, step, expected_messages_list):
        fake_model = MagicMock()
        agent = CodeAgent(
            tools=[],
            model=fake_model,
        )
        task = "Test task"
        agent.planning_step(task, is_first_step=(step == 1), step=step)
        expected_message_texts = {
            "INITIAL_PLAN_USER_PROMPT": populate_template(
                agent.prompt_templates["planning"]["initial_plan"],
                variables=dict(
                    task=task,
                    tools=agent.tools,
                    managed_agents=agent.managed_agents,
                    answer_facts=agent.memory.steps[0].model_output_message.content,
                ),
            ),
            "UPDATE_PLAN_SYSTEM_PROMPT": populate_template(
                agent.prompt_templates["planning"]["update_plan_pre_messages"], variables=dict(task=task)
            ),
            "UPDATE_PLAN_USER_PROMPT": populate_template(
                agent.prompt_templates["planning"]["update_plan_post_messages"],
                variables=dict(
                    task=task,
                    tools=agent.tools,
                    managed_agents=agent.managed_agents,
                    facts_update=agent.memory.steps[0].model_output_message.content,
                    remaining_steps=agent.max_steps - step,
                ),
            ),
        }
        for expected_messages in expected_messages_list:
            for expected_message in expected_messages:
                for expected_content in expected_message["content"]:
                    expected_content["text"] = expected_message_texts[expected_content["text"]]
        assert len(agent.memory.steps) == 1
        planning_step = agent.memory.steps[0]
        assert isinstance(planning_step, PlanningStep)
        expected_model_input_messages = expected_messages_list[0]
        model_input_messages = planning_step.model_input_messages
        assert isinstance(model_input_messages, list)
        assert len(model_input_messages) == len(expected_model_input_messages)  # 2
        for message, expected_message in zip(model_input_messages, expected_model_input_messages):
            assert isinstance(message, dict)
            assert "role" in message
            assert "content" in message
            assert message["role"] in MessageRole.__members__.values()
            assert message["role"] == expected_message["role"]
            assert isinstance(message["content"], list)
            assert len(message["content"]) == 1
            for content, expected_content in zip(message["content"], expected_message["content"]):
                assert content == expected_content
        # Test calls to model
        assert len(fake_model.call_args_list) == 1
        for call_args, expected_messages in zip(fake_model.call_args_list, expected_messages_list):
            assert len(call_args.args) == 1
            messages = call_args.args[0]
            assert isinstance(messages, list)
            assert len(messages) == len(expected_messages)
            for message, expected_message in zip(messages, expected_messages):
                assert isinstance(message, dict)
                assert "role" in message
                assert "content" in message
                assert message["role"] in MessageRole.__members__.values()
                assert message["role"] == expected_message["role"]
                assert isinstance(message["content"], list)
                assert len(message["content"]) == 1
                for content, expected_content in zip(message["content"], expected_message["content"]):
                    assert content == expected_content

    @pytest.mark.parametrize(
        "images, expected_messages_list",
        [
            (
                None,
                [
                    [
                        {
                            "role": MessageRole.SYSTEM,
                            "content": [{"type": "text", "text": "FINAL_ANSWER_SYSTEM_PROMPT"}],
                        },
                        {"role": MessageRole.USER, "content": [{"type": "text", "text": "FINAL_ANSWER_USER_PROMPT"}]},
                    ]
                ],
            ),
            (
                ["image1.png"],
                [
                    [
                        {
                            "role": MessageRole.SYSTEM,
                            "content": [{"type": "text", "text": "FINAL_ANSWER_SYSTEM_PROMPT"}, {"type": "image"}],
                        },
                        {"role": MessageRole.USER, "content": [{"type": "text", "text": "FINAL_ANSWER_USER_PROMPT"}]},
                    ]
                ],
            ),
        ],
    )
    def test_provide_final_answer(self, images, expected_messages_list):
        fake_model = MagicMock()
        fake_model.return_value.content = "Final answer."
        agent = CodeAgent(
            tools=[],
            model=fake_model,
        )
        task = "Test task"
        final_answer = agent.provide_final_answer(task, images=images)
        expected_message_texts = {
            "FINAL_ANSWER_SYSTEM_PROMPT": agent.prompt_templates["final_answer"]["pre_messages"],
            "FINAL_ANSWER_USER_PROMPT": populate_template(
                agent.prompt_templates["final_answer"]["post_messages"], variables=dict(task=task)
            ),
        }
        for expected_messages in expected_messages_list:
            for expected_message in expected_messages:
                for expected_content in expected_message["content"]:
                    if "text" in expected_content:
                        expected_content["text"] = expected_message_texts[expected_content["text"]]
        assert final_answer == "Final answer."
        # Test calls to model
        assert len(fake_model.call_args_list) == 1
        for call_args, expected_messages in zip(fake_model.call_args_list, expected_messages_list):
            assert len(call_args.args) == 1
            messages = call_args.args[0]
            assert isinstance(messages, list)
            assert len(messages) == len(expected_messages)
            for message, expected_message in zip(messages, expected_messages):
                assert isinstance(message, dict)
                assert "role" in message
                assert "content" in message
                assert message["role"] in MessageRole.__members__.values()
                assert message["role"] == expected_message["role"]
                assert isinstance(message["content"], list)
                assert len(message["content"]) == len(expected_message["content"])
                for content, expected_content in zip(message["content"], expected_message["content"]):
                    assert content == expected_content

    @pytest.mark.parametrize(
        "tools, managed_agents, name, expectation",
        [
            # Valid case: no duplicates
            (
                [MockTool("tool1"), MockTool("tool2")],
                [MockAgent("agent1", [MockTool("tool3")])],
                "test_agent",
                does_not_raise(),
            ),
            # Invalid case: duplicate tool names
            ([MockTool("tool1"), MockTool("tool1")], [], "test_agent", pytest.raises(ValueError)),
            # Invalid case: tool name same as managed agent name
            (
                [MockTool("tool1")],
                [MockAgent("tool1", [MockTool("final_answer")])],
                "test_agent",
                pytest.raises(ValueError),
            ),
            # Valid case: tool name same as managed agent's tool name
            ([MockTool("tool1")], [MockAgent("agent1", [MockTool("tool1")])], "test_agent", does_not_raise()),
            # Invalid case: duplicate managed agent name and managed agent tool name
            ([MockTool("tool1")], [], "tool1", pytest.raises(ValueError)),
            # Valid case: duplicate tool names across managed agents
            (
                [MockTool("tool1")],
                [
                    MockAgent("agent1", [MockTool("tool2"), MockTool("final_answer")]),
                    MockAgent("agent2", [MockTool("tool2"), MockTool("final_answer")]),
                ],
                "test_agent",
                does_not_raise(),
            ),
        ],
    )
    def test_validate_tools_and_managed_agents(self, tools, managed_agents, name, expectation):
        fake_model = MagicMock()
        with expectation:
            MultiStepAgent(
                tools=tools,
                model=fake_model,
                name=name,
                managed_agents=managed_agents,
            )


class TestToolCallingAgent(unittest.TestCase):
    @patch("huggingface_hub.InferenceClient")
    def test_toolcalling_agent_api(self, mock_inference_client):
        mock_client = mock_inference_client.return_value
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[
            0
        ].message.content = '{"name": "weather_api", "arguments": {"location": "Paris", "date": "today"}}'
        mock_client.chat_completion.return_value = mock_response
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20

        model = HfApiModel(model_id="test-model")

        from smolagents import tool

        @tool
        def weather_api(location: str, date: str) -> str:
            """
            Gets the weather in the next days at given location.
            Args:
                location: the location
                date: the date
            """
            return f"The weather in {location} on date:{date} is sunny."

        agent = ToolCallingAgent(model=model, tools=[weather_api], max_steps=1)
        agent.run("What's the weather in Paris?")
        assert agent.memory.steps[0].task == "What's the weather in Paris?"
        assert agent.memory.steps[1].tool_calls[0].name == "weather_api"
        assert agent.memory.steps[1].tool_calls[0].arguments == {"location": "Paris", "date": "today"}
        assert agent.memory.steps[1].observations == "The weather in Paris on date:today is sunny."

        mock_response.choices[0].message.tool_calls = [
            ChatMessageToolCall(
                function=ChatMessageToolCallDefinition(
                    name="weather_api", arguments='{"location": "Paris", "date": "today"}'
                ),
                id="call_0",
                type="function",
            )
        ]
        mock_response.choices[0].message.content = None
        mock_client.chat_completion.return_value = mock_response

        agent.run("What's the weather in Paris?")
        assert agent.memory.steps[0].task == "What's the weather in Paris?"
        assert agent.memory.steps[1].tool_calls[0].name == "weather_api"
        assert agent.memory.steps[1].tool_calls[0].arguments == {"location": "Paris", "date": "today"}
        assert agent.memory.steps[1].observations == "The weather in Paris on date:today is sunny."


class TestCodeAgent:
    @pytest.mark.parametrize("provide_run_summary", [False, True])
    def test_call_with_provide_run_summary(self, provide_run_summary):
        agent = CodeAgent(tools=[], model=MagicMock(), provide_run_summary=provide_run_summary)
        assert agent.provide_run_summary is provide_run_summary
        agent.managed_agent_prompt = "Task: {task}"
        agent.name = "test_agent"
        agent.run = MagicMock(return_value="Test output")
        agent.write_memory_to_messages = MagicMock(return_value=[{"content": "Test summary"}])

        result = agent("Test request")
        expected_summary = "Here is the final answer from your managed agent 'test_agent':\nTest output"
        if provide_run_summary:
            expected_summary += (
                "\n\nFor more detail, find below a summary of this agent's work:\n"
                "<summary_of_work>\n\nTest summary\n---\n</summary_of_work>"
            )
        assert result == expected_summary

    def test_errors_logging(self):
        def fake_code_model(messages, stop_sequences=None, grammar=None) -> str:
            return ChatMessage(role="assistant", content="Code:\n```py\nsecret=3;['1', '2'][secret]\n```")

        agent = CodeAgent(tools=[], model=fake_code_model, verbosity_level=1)

        with agent.logger.console.capture() as capture:
            agent.run("Test request")
        assert "secret\\\\" in repr(capture.get())

    def test_missing_import_triggers_advice_in_error_log(self):
        # Set explicit verbosity level to 1 to override the default verbosity level of -1 set in CI fixture
        agent = CodeAgent(tools=[], model=fake_code_model_import, verbosity_level=1)

        with agent.logger.console.capture() as capture:
            agent.run("Count to 3")
        str_output = capture.get()
        assert "`additional_authorized_imports`" in str_output.replace("\n", "")

    def test_errors_show_offending_line_and_error(self):
        agent = CodeAgent(tools=[PythonInterpreterTool()], model=fake_code_model_error)
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, AgentText)
        assert output == "got an error"
        assert "Code execution failed at line 'error_function()'" in str(agent.memory.steps[1].error)
        assert "ValueError" in str(agent.memory.steps)

    def test_error_saves_previous_print_outputs(self):
        agent = CodeAgent(tools=[PythonInterpreterTool()], model=fake_code_model_error, verbosity_level=10)
        agent.run("What is 2 multiplied by 3.6452?")
        assert "Flag!" in str(agent.memory.steps[1].observations)

    def test_syntax_error_show_offending_lines(self):
        agent = CodeAgent(tools=[PythonInterpreterTool()], model=fake_code_model_syntax_error)
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, AgentText)
        assert output == "got an error"
        assert '    print("Failing due to unexpected indent")' in str(agent.memory.steps)

    def test_end_code_appending(self):
        # Checking original output message
        orig_output = fake_code_model_no_return([])
        assert not orig_output.content.endswith("<end_code>")

        # Checking the step output
        agent = CodeAgent(
            tools=[PythonInterpreterTool()],
            model=fake_code_model_no_return,
            max_steps=1,
        )
        answer = agent.run("What is 2 multiplied by 3.6452?")
        assert answer

        memory_steps = agent.memory.steps
        actions_steps = [s for s in memory_steps if isinstance(s, ActionStep)]

        outputs = [s.model_output for s in actions_steps if s.model_output]
        assert outputs
        assert all(o.endswith("<end_code>") for o in outputs)

        messages = [s.model_output_message for s in actions_steps if s.model_output_message]
        assert messages
        assert all(m.content.endswith("<end_code>") for m in messages)

    def test_change_tools_after_init(self):
        from smolagents import tool

        @tool
        def fake_tool_1() -> str:
            """Fake tool"""
            return "1"

        @tool
        def fake_tool_2() -> str:
            """Fake tool"""
            return "2"

        def fake_code_model(messages, stop_sequences=None, grammar=None) -> str:
            return ChatMessage(role="assistant", content="Code:\n```py\nfinal_answer(fake_tool_1())\n```")

        agent = CodeAgent(tools=[fake_tool_1], model=fake_code_model)

        agent.tools["final_answer"] = CustomFinalAnswerTool()
        agent.tools["fake_tool_1"] = fake_tool_2

        answer = agent.run("Fake task.")
        assert answer == "2CUSTOM"

    @pytest.mark.parametrize("agent_dict_version", ["v1.9", "v1.10"])
    def test_from_folder(self, agent_dict_version, get_agent_dict):
        agent_dict = get_agent_dict(agent_dict_version)
        with patch("smolagents.agents.Path") as mock_path, patch("smolagents.models.HfApiModel") as mock_model:
            import json

            mock_path.return_value.__truediv__.return_value.read_text.return_value = json.dumps(agent_dict)
            mock_model.from_dict.return_value.model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
            agent = CodeAgent.from_folder("ignored_dummy_folder")
        assert isinstance(agent, CodeAgent)
        assert agent.name == "test_agent"
        assert agent.description == "dummy description"
        assert agent.max_steps == 10
        assert agent.planning_interval == 2
        assert agent.grammar is None
        assert agent.additional_authorized_imports == ["pandas"]
        assert "pandas" in agent.authorized_imports
        assert agent.executor_type == "local"
        assert agent.executor_kwargs == {}
        assert agent.max_print_outputs_length is None
        assert agent.managed_agents == {}
        assert set(agent.tools.keys()) == {"final_answer"}
        assert agent.model == mock_model.from_dict.return_value
        assert mock_model.from_dict.call_args.args[0]["model_id"] == "Qwen/Qwen2.5-Coder-32B-Instruct"
        assert agent.model.model_id == "Qwen/Qwen2.5-Coder-32B-Instruct"
        assert agent.logger.level == 2
        assert agent.prompt_templates["system_prompt"] == "dummy system prompt"


class TestMultiAgents:
    def test_multiagents_save(self, tmp_path):
        model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", max_tokens=2096, temperature=0.5)

        web_agent = ToolCallingAgent(
            model=model,
            tools=[DuckDuckGoSearchTool(max_results=2), VisitWebpageTool()],
            name="web_agent",
            description="does web searches",
        )
        code_agent = CodeAgent(model=model, tools=[], name="useless", description="does nothing in particular")

        agent = CodeAgent(
            model=model,
            tools=[],
            additional_authorized_imports=["pandas", "datetime"],
            managed_agents=[web_agent, code_agent],
            max_print_outputs_length=1000,
            executor_type="local",
            executor_kwargs={"max_workers": 2},
        )
        agent.save(tmp_path)

        expected_structure = {
            "managed_agents": {
                "useless": {"tools": {"files": ["final_answer.py"]}, "files": ["agent.json", "prompts.yaml"]},
                "web_agent": {
                    "tools": {"files": ["final_answer.py", "visit_webpage.py", "web_search.py"]},
                    "files": ["agent.json", "prompts.yaml"],
                },
            },
            "tools": {"files": ["final_answer.py"]},
            "files": ["app.py", "requirements.txt", "agent.json", "prompts.yaml"],
        }

        def verify_structure(current_path: Path, structure: dict):
            for dir_name, contents in structure.items():
                if dir_name != "files":
                    # For directories, verify they exist and recurse into them
                    dir_path = current_path / dir_name
                    assert dir_path.exists(), f"Directory {dir_path} does not exist"
                    assert dir_path.is_dir(), f"{dir_path} is not a directory"
                    verify_structure(dir_path, contents)
                else:
                    # For files, verify each exists in the current path
                    for file_name in contents:
                        file_path = current_path / file_name
                        assert file_path.exists(), f"File {file_path} does not exist"
                        assert file_path.is_file(), f"{file_path} is not a file"

        verify_structure(tmp_path, expected_structure)

        # Test that re-loaded agents work as expected.
        agent2 = CodeAgent.from_folder(tmp_path, planning_interval=5)
        assert agent2.planning_interval == 5  # Check that kwargs are used
        assert set(agent2.authorized_imports) == set(["pandas", "datetime"] + BASE_BUILTIN_MODULES)
        assert agent2.max_print_outputs_length == 1000
        assert agent2.executor_type == "local"
        assert agent2.executor_kwargs == {"max_workers": 2}
        assert (
            agent2.managed_agents["web_agent"].tools["web_search"].max_results == 10
        )  # For now tool init parameters are forgotten
        assert agent2.model.kwargs["temperature"] == pytest.approx(0.5)

    def test_multiagents(self):
        class FakeModelMultiagentsManagerAgent:
            model_id = "fake_model"

            def __call__(
                self,
                messages,
                stop_sequences=None,
                grammar=None,
                tools_to_call_from=None,
            ):
                if tools_to_call_from is not None:
                    if len(messages) < 3:
                        return ChatMessage(
                            role="assistant",
                            content="",
                            tool_calls=[
                                ChatMessageToolCall(
                                    id="call_0",
                                    type="function",
                                    function=ChatMessageToolCallDefinition(
                                        name="search_agent",
                                        arguments="Who is the current US president?",
                                    ),
                                )
                            ],
                        )
                    else:
                        assert "Report on the current US president" in str(messages)
                        return ChatMessage(
                            role="assistant",
                            content="",
                            tool_calls=[
                                ChatMessageToolCall(
                                    id="call_0",
                                    type="function",
                                    function=ChatMessageToolCallDefinition(
                                        name="final_answer", arguments="Final report."
                                    ),
                                )
                            ],
                        )
                else:
                    if len(messages) < 3:
                        return ChatMessage(
                            role="assistant",
                            content="""
Thought: Let's call our search agent.
Code:
```py
result = search_agent("Who is the current US president?")
```<end_code>
""",
                        )
                    else:
                        assert "Report on the current US president" in str(messages)
                        return ChatMessage(
                            role="assistant",
                            content="""
Thought: Let's return the report.
Code:
```py
final_answer("Final report.")
```<end_code>
""",
                        )

        manager_model = FakeModelMultiagentsManagerAgent()

        class FakeModelMultiagentsManagedAgent:
            model_id = "fake_model"

            def __call__(
                self,
                messages,
                tools_to_call_from=None,
                stop_sequences=None,
                grammar=None,
            ):
                return ChatMessage(
                    role="assistant",
                    content="Here is the secret content: FLAG1",
                    tool_calls=[
                        ChatMessageToolCall(
                            id="call_0",
                            type="function",
                            function=ChatMessageToolCallDefinition(
                                name="final_answer",
                                arguments="Report on the current US president",
                            ),
                        )
                    ],
                )

        managed_model = FakeModelMultiagentsManagedAgent()

        web_agent = ToolCallingAgent(
            tools=[],
            model=managed_model,
            max_steps=10,
            name="search_agent",
            description="Runs web searches for you. Give it your request as an argument. Make the request as detailed as needed, you can ask for thorough reports",
            verbosity_level=2,
        )

        manager_code_agent = CodeAgent(
            tools=[],
            model=manager_model,
            managed_agents=[web_agent],
            additional_authorized_imports=["time", "numpy", "pandas"],
        )

        report = manager_code_agent.run("Fake question.")
        assert report == "Final report."

        manager_toolcalling_agent = ToolCallingAgent(
            tools=[],
            model=manager_model,
            managed_agents=[web_agent],
        )

        with web_agent.logger.console.capture() as capture:
            report = manager_toolcalling_agent.run("Fake question.")
        assert report == "Final report."
        assert "FLAG1" in capture.get()  # Check that managed agent's output is properly logged

        # Test that visualization works
        with manager_toolcalling_agent.logger.console.capture() as capture:
            manager_toolcalling_agent.visualize()
        assert "├──" in capture.get()


@pytest.fixture
def prompt_templates():
    return {
        "system_prompt": "This is a test system prompt.",
        "managed_agent": {"task": "Task for {{name}}: {{task}}", "report": "Report for {{name}}: {{final_answer}}"},
    }
