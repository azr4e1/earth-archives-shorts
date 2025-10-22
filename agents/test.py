from openai_agent import WriterAgent, ChunkerAgent, VeoPrompter
from elevenlabs_agents import AudioGenerationAgent
from video_generation import VideoGenerationAgent
from utils import Cacher
import asyncio
import json

query = 'What are the intelligent species of the Helionis Cluster?'
writer = WriterAgent('Writer', 'gpt-4.1',
                     vector_store_id="vs_68f01ec9d8a08191b2ace026d2cf8a80")
chunker = ChunkerAgent('Chunker', 'gpt-4.1')

# TODO: recovery: provide a flag for each stage to a path where everything gets saved.
# depending on where the worflow is interrupted, we can recover from that point by looking into the
# save directory


async def main():
    cacher = Cacher()
    script = await writer.run(query=query)
    cacher.save_script(script)

    chunks_model = await chunker.run(script=script)
    chunks = chunks_model.model_dump()['descriptions']
    cacher.save_chunks(chunks)

    semaphore = asyncio.Semaphore(3)
    audio_agents = [AudioGenerationAgent(f"AudioGeneration_{i}",
                                         "nrbjbLmJZ7T1FcsFbbeE",
                                         "eleven_multilingual_v2",
                                         semaphore=semaphore,
                                         settings={'speed': 1.1})
                    for i in range(len(chunks))]
    audios = await asyncio.gather(*[audio_agents[i].run(chunks[i])
                                  for i in range(len(chunks))])
    cacher.save_audio(audios)

    veo_prompters = [VeoPrompter(
        f'Prompter_{i}', 'gpt-4.1') for i in range(len(chunks))]
    lengths = [len(audio) / 1000 for audio in audios]
    n_descriptions = [l // 8 if l % 8 < 4 else (l // 8) + 1 for l in lengths]
    result = await asyncio.gather(*[veo_prompters[i].run(chunks[i], n_descriptions[i])
                                    for i in range(len(chunks))])

    descriptions = {}
    for i, r in enumerate(result):
        descriptions[chunks[i]] = r.model_dump()['descriptions']

    cacher.save_descriptions(descriptions)

    descriptions_flat = [d for _, descs in descriptions.items() for d in descs]
    video_semaphore = asyncio.Semaphore(3)
    video_agents = [VideoGenerationAgent(f"VideoGeneration_{i}",
                                         semaphore=video_semaphore,
                                         settings={"aspectRatio": "9:16",
                                                   "durationSeconds": "8"})
                    for i in range(len(descriptions_flat))]
    videos = await asyncio.gather(*[video_agents[i].run(desc)
                                  for i, desc in enumerate(descriptions_flat)])
    cacher.save_videos(videos)

asyncio.run(main())
