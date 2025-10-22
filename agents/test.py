from openai_agent import WriterAgent, ChunkerAgent, VeoPrompter
from elevenlabs_agents import VoiceGenerationAgent
from video_generation import VideoGenerationAgent
from utils import Cacher, generate_hash
import asyncio

ELEVENLABS_SEMAPHORE = 3
VEO_SEMAPHORE = 3

# TODO: recovery: provide a flag for each stage to a path where everything gets saved.
# depending on where the worflow is interrupted, we can recover from that point by looking into the
# save directory


async def process_script(query):
    writer = WriterAgent('Writer', 'gpt-4.1',
                         vector_store_id="vs_68f01ec9d8a08191b2ace026d2cf8a80")
    script = await writer.run(query=query)
    return script


async def process_chunks(script):
    chunker = ChunkerAgent('Chunker', 'gpt-4.1')
    chunks_model = await chunker.run(script=script)
    chunks = chunks_model.model_dump()['descriptions']

    return chunks


async def process_voice(chunks):
    chunks_l = len(chunks)
    semaphore = asyncio.Semaphore(ELEVENLABS_SEMAPHORE)
    audio_agents = [VoiceGenerationAgent(f"AudioGeneration_{i}",
                                         "nrbjbLmJZ7T1FcsFbbeE",
                                         "eleven_multilingual_v2",
                                         semaphore=semaphore,
                                         settings={'speed': 1.1})
                    for i in range(chunks_l)]
    audios_output = await asyncio.gather(*[audio_agents[i].run(chunks[i])
                                           for i in range(chunks_l)])
    audios = {generate_hash(chunks[i]): audios_output[i]
              for i in range(chunks_l)}

    return audios


async def process_veo_prompts(chunks, audios):
    chunks_l = len(chunks)
    veo_prompters = [VeoPrompter(
        f'Prompter_{i}', 'gpt-4.1') for i in range(chunks_l)]
    lengths = [len(audio) / 1000 for audio in audios.values()]
    n_descriptions = [l // 8 if l % 8 < 4 else (l // 8) + 1 for l in lengths]
    result = await asyncio.gather(*[veo_prompters[i].run(chunks[i], n_descriptions[i])
                                    for i in range(chunks_l)])

    descriptions = {}
    for i, r in enumerate(result):
        descriptions[chunks[i]] = r.model_dump()['descriptions']

    return descriptions


async def process_video(descriptions):
    video_semaphore = asyncio.Semaphore(VEO_SEMAPHORE)
    descriptions_flat = [d for _, descs in descriptions.items() for d in descs]
    video_agents = [VideoGenerationAgent(f"VideoGeneration_{i}",
                                         semaphore=video_semaphore,
                                         settings={"aspectRatio": "9:16",
                                                   "durationSeconds": "8"})
                    for i in range(len(descriptions_flat))]
    video_output = await asyncio.gather(*[video_agents[i].run(desc)
                                          for i, desc in enumerate(descriptions_flat)])
    video_names = []
    for chunk, descs in descriptions.items():
        hashname = generate_hash(chunk)
        for i in range(len(descs)):
            video_names.append(f"{hashname}_{i}")

    videos = {video_names[i]: video_output[i]
              for i in range(len(descriptions_flat))}

    return videos


async def main():
    cacher = Cacher()
    query = 'Write a script about the Onyx Hive'

    script = await process_script(query)
    cacher.save_script(script)

    chunks = await process_chunks(script)
    cacher.save_chunks(chunks)

    audios = await process_voice(chunks)
    cacher.save_audio(audios)

    descriptions = await process_veo_prompts(chunks, audios)
    cacher.save_descriptions(descriptions)

    videos = await process_video(descriptions)
    cacher.save_videos(videos)

asyncio.run(main())
