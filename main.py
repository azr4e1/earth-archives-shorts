from agents import FileSearchTool, Agent, ModelSettings, TResponseInputItem, Runner, RunConfig
import sys
import asyncio
from pydantic import BaseModel
from datetime import datetime
import json
from pathlib import Path
import aiofiles
from elevenlabs.client import AsyncElevenLabs
from dotenv import load_dotenv
import os
from pydub import AudioSegment
import io
import openai
from google.genai import types
from google import genai

load_dotenv()

VEO_LENGTHS = [4, 6, 8]


elevenlabs = AsyncElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
)

# Tool definitions
file_search = FileSearchTool(
    vector_store_ids=[
        "vs_68f01ec9d8a08191b2ace026d2cf8a80"
    ]
)


class ChunkerSchema(BaseModel):
    descriptions: list[str]


def generate_video(client, prompt, filename):
    model_names = ["veo-3.0-generate-001",
                   "veo-3.0-fast-generate-001", "veo-2.0-generate-001"]
    for model in model_names:
        try:
            operation = client.models.generate_videos(
                model=model,
                prompt=prompt,
                config=types.GenerateVideosConfig(
                    aspectRatio="9:16", durationSeconds="8")
            )

            # Poll the operation status until the video is ready.
            while not operation.done:
                operation = client.operations.get(operation)

            generated_video = operation.response.generated_videos[0]
            client.files.download(file=generated_video.video)
            generated_video.video.save(filename)
            return "Success"
        except Exception as e:
            print(e)
            continue

    return "Failure"


writer = Agent(
    name="Writer",
    instructions="""You are a skilled screen writer and narrator. You have create a fictional universe called the Earth Archives, accessible through a vector store. Take the input, and together with your knowledge of The Earth Archives universe (from the vector store), create a script to voice over a video of about 1 minutes.

The script must be long enough to cover the 1 minutes length requirement. It must not have bullet points nor headers or titles. It must read like a novel of Frank Herbert, it must flow and be pleasant to listen to.""",
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


async def veo_descriptor(client: openai.AsyncOpenAI, input: str, versions: int, model: str) -> ChunkerSchema:
    system_prompt = """You are a Veo3 expert user. Your expertise lies in being able to craft perfect prompts for video generation generative models.

Your 'input' variable is a script narrating over something, and the number of different prompts you have to write. Your job is to rewrite the script as many times as specified by the variable 'versions' so that they are the perfect prompts to be used by a video generation model like veo3 or sora, so that they generate a clip to accompany that part of the narration.

For science fiction words and objects that do not exist in our world, create a full description that clearly explains to the model what it needs to show. You have full creative liberty in how to describe things that are not explained.

Descriptions of creatures and worlds must stay consistent between versions.

Decide in advance the style of the video and apply it consistently to all versions. Include lighting and shot frame and motion.

The following elements should be included in your prompt:

    * Subject: The object, person, animal, or scenery that you want in your video, such as cityscape, nature, vehicles, or puppies.
    * Action: What the subject is doing (for example, walking, running, or turning their head).
    * Style: Specify creative direction using specific film style keywords, such as sci-fi, horror film, film noir, or animated styles like cartoon.
    * Camera positioning and motion: [Optional] Control the camera's location and movement using terms like aerial view, eye-level, top-down shot, dolly shot, or worms eye.
    * Composition: [Optional] How the shot is framed, such as wide shot, close-up, single-shot or two-shot.
    * Focus and lens effects: [Optional] Use terms like shallow focus, deep focus, soft focus, macro lens, and wide-angle lens to achieve specific visual effects.
    * Ambiance: [Optional] How the color and light contribute to the scene, such as blue tones, night, or warm tones.

More tips for writing prompts:
    * Use descriptive language: Use adjectives and adverbs to paint a clear picture for Veo.

Example: Historical diplomatic chamber, crystalline lantern lighting: A slow orbital camera movement circles around a formal assembly of Hathari diplomats—tall, elegant beings with mirrored silver skin that catches and reflects the cool, precise light. They wear delicately layered robes in flowing, translucent fabrics that shimmer subtly with each composed gesture. The Hathari stand in a semicircle formation, their postures serene and deliberate as they negotiate terms in an interstellar council session. One diplomat extends their hand in a measured gesture of agreement, fingers spreading gracefully. Another inclines their head, the motion slow and weighted with significance. A third diplomat's robe catches the light, revealing intricate patterns woven into the fabric. The camera moves from a medium wide shot to a medium close-up on a lead diplomat's face, capturing the subtle expressiveness in their features—a slight narrowing of the eyes, a barely perceptible shift in their reflective skin tone. Audio: Soft ambient hum of chamber acoustics, quiet rustling of fabric, low murmur of diplomatic discourse in an unknown language. The lighting remains consistently cool-toned—pale blues and silvers—highlighting the reflective surfaces of their garments and skin, creating an atmosphere of formality, precision, and measured diplomacy.

Exclude all science fiction words that do not belong to common language, and replace them with your description in common language. You have full creative liberty in this.
"""

    user_template = f"""
***INPUT***
{input}

***VERSIONS***
{versions}
"""
    response = await client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_template,
            },
        ],
        text_format=ChunkerSchema,
    )

    return response.output_parsed

chunker = Agent(
    name="Chunker",
    instructions="""You are a semantic expert. Your expertise lies in being able to identify the transitions moments in a script where the topic changes, slightly or significantly. In the context of a movie script, you are able to split the script in chunks that can be used as references to create clips that, when put together, create the final video the script will be voiced over.

Your input is a script narrating over something. Your job is to split the script into a number of chunks as described earlier. DO NOW REWRITE ANY PART OF THE SCRIPT. EACH CHUNK MUST BE A SLICE OF THE SCRIPT AS IS, WITH NO MODIFICATION WHATSOEVER, such that, when putting together all the chunks, we get the original script unaltered.""",
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


async def elevenlabs_generation(input: str, name: str, semaphore):
    async with semaphore:
        audio = elevenlabs.text_to_speech.convert(
            text=input,
            voice_id="nrbjbLmJZ7T1FcsFbbeE",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )

        buffer = bytearray()
        Path("./audio").mkdir(exist_ok=True)
        async with aiofiles.open(f"./audio/{name}.mp3", "wb") as f:
            async for chunk in audio:
                await f.write(chunk)
                buffer.extend(chunk)

        return buffer


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
    semaphore = asyncio.Semaphore(3)
    script, chunks = await run_workflow(input)
    chunks['full_script'] = script
    Path("./scripts").mkdir(exist_ok=True)
    with open(f"./scripts/{datetime.now().strftime('%Y%m%d%H%M')}-description.json", "w") as f:
        json.dump(chunks, f)

    # generate audio tracks
    tasks = [elevenlabs_generation(desc, i, semaphore) for i, desc in enumerate(
        chunks['descriptions'])]
    audios = await asyncio.gather(*tasks)

    # determine length of each track
    lengths = [len(AudioSegment.from_mp3(io.BytesIO(
        audio))) / 1000 for audio in audios]
    n_descriptions = [l // 8 if l % 8 < 4 else (l // 8) + 1 for l in lengths]

    client = openai.AsyncOpenAI()
    model = "gpt-4.1"
    tasks = [veo_descriptor(client, chunk, n_descriptions[i], model)
             for i, chunk in enumerate(chunks['descriptions'])]
    desc_output = await asyncio.gather(*tasks)
    descriptions = {chunk: desc_output[i].model_dump()['descriptions']
                    for i, chunk in enumerate(chunks['descriptions'])}
    with open("./descriptions.json", "w") as f:
        json.dump(descriptions, f)

    client = genai.Client()
    descs = [d for _, descs in descriptions.items() for d in descs]
    for i, d in enumerate(descs):
        name = f"video_{i}.mp4"
        generate_video(client, d, name)


if __name__ == "__main__":
    input = sys.argv[1]
    asyncio.run(main(WorkflowInput(input_as_text=input)))
