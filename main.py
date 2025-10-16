from agents import FileSearchTool, Agent, ModelSettings, TResponseInputItem, Runner, RunConfig
import sys
import asyncio
from pydantic import BaseModel
from datetime import datetime
import json

# Tool definitions
file_search = FileSearchTool(
    vector_store_ids=[
        "vs_68f01ec9d8a08191b2ace026d2cf8a80"
    ]
)


class ChunkerSchema(BaseModel):
    descriptions: list[str]


writer = Agent(
    name="Writer",
    instructions="""You are a skilled screen writer and narrator. You have create a fictional universe called the Earth Archives, accessible through a vector store. Take the input, and together with your knowledge of The Earth Archives universe (from the vector store), create a script to voice over a video of about 2 minutes.

The script must be long enough to cover the 2 minutes length requirement. It must not have bullet points nor headers or titles. It must read like a speech of David Attenborough, it must flow and be pleasant to listen to.""",
    model="gpt-4.1",
    tools=[
        file_search
    ],
    model_settings=ModelSettings(
        temperature=1,
        top_p=1,
        max_tokens=16385,
        store=True
    )
)


chunker = Agent(
    name="Chunker",
    instructions="""You are a Veo3 expert user. Your expertise lies in being able to craft perfect prompts for video generation generative models.

Your input is a script narrating over something. Your job is to split the script into a number of chunks, and rewrite those chunks so that they are the perfect prompt to be used by a video generation model like veo3 or sora, so that they generate a clip to accompany that part of the narration.

Each chunk must be self sufficient, so you cannot use words like \"then\" or \"after\". Each description must be able to be taken independently and still create an amazing video without further context.

For science fiction words and objects that do not exist in our world, create a full description that clearly explains to the model what it needs to show. You have full creative liberty in how to describe things that are not explained.

Decide in advance the style of the video and apply it consistently to all descriptions. Include lighting and shot frame and motion.

Example: A medium shot, historical adventure setting: Warm lamplight illuminates a cartographer in a cluttered study, poring over an ancient, sprawling map spread across a large table.""",
    model="gpt-4.1",
    output_type=ChunkerSchema,
    model_settings=ModelSettings(
        temperature=1,
        top_p=1,
        max_tokens=16385,
        store=True
    )
)


class WorkflowInput(BaseModel):
    input_as_text: str


# Main code entrypoint
async def run_workflow(workflow_input: WorkflowInput):
    state = {

    }
    workflow = workflow_input.model_dump()
    conversation_history: list[TResponseInputItem] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": workflow["input_as_text"]
                }
            ]
        }
    ]
    writer_result_temp = await Runner.run(
        writer,
        input=[
            *conversation_history
        ],
        run_config=RunConfig(trace_metadata={
            "__trace_source__": "agent-builder",
            "workflow_id": "wf_68f02184b74c8190ae10eaa7caf2992a0702b777659adb87"
        })
    )

    conversation_history.extend([item.to_input_item()
                                for item in writer_result_temp.new_items])
    # print("Writer:\n"+20*"=")
    # for item in writer_result_temp.new_items:
    #     print(item)

    writer_result = {
        "output_text": writer_result_temp.final_output_as(str)
    }
    chunker_result_temp = await Runner.run(
        chunker,
        input=[
            *conversation_history
        ],
        run_config=RunConfig(trace_metadata={
            "__trace_source__": "agent-builder",
            "workflow_id": "wf_68f02184b74c8190ae10eaa7caf2992a0702b777659adb87"
        })
    )

    conversation_history.extend([item.to_input_item()
                                for item in chunker_result_temp.new_items])
    # print("Chunker:\n"+20*"=")
    # for item in chunker_result_temp.new_items:
    #     print(item)

    chunker_result = {
        "output_text": chunker_result_temp.final_output.model_dump_json(),
        "output_parsed": chunker_result_temp.final_output.model_dump()
    }

    return writer_result['output_text'], chunker_result['output_parsed']


async def main(input):
    script, descriptions = await run_workflow(input)
    descriptions['script'] = script
    with open(f"./scripts/{datetime.now().strftime('%Y%m%d%H%M')}-description.json", "w") as f:
        json.dump(descriptions, f)


if __name__ == "__main__":
    input = sys.argv[1]
    asyncio.run(main(WorkflowInput(input_as_text=input)))
