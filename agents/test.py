from openai_agent import WriterAgent, ChunkerAgent, VeoPrompter
from elevenlabs_agents import AudioGenerationAgent
import asyncio
import json

query = 'What are the intelligent species of the Helionis Cluster?'
writer = WriterAgent('Writer', 'gpt-4.1',
                     vector_store_id="vs_68f01ec9d8a08191b2ace026d2cf8a80")
chunker = ChunkerAgent('Chunker', 'gpt-4.1')

# logging.basicConfig(level=logging.DEBUG)


async def main():
    script = await writer.run(query=query)
    chunks = await chunker.run(script=script)

    chunks = chunks.model_dump()['descriptions']
    semaphore = asyncio.Semaphore(3)
    audio_agents = [AudioGenerationAgent(f"AudioGeneration_{
                                         i}", "nrbjbLmJZ7T1FcsFbbeE", "eleven_multilingual_v2", semaphore=semaphore, settings={'speed': 1.1}) for i in range(len(chunks))]
    audios = await asyncio.gather(*[audio_agents[i].run(chunks[i]) for i in range(len(chunks))])

    for i, audio in enumerate(audios):
        audio.export(f"audio_{i}.mp3", format='mp3')

    veo_prompters = [VeoPrompter(
        f'Prompter_{i}', 'gpt-4.1') for i in range(len(chunks))]
    lengths = [len(audio) / 1000 for audio in audios]
    n_descriptions = [l // 8 if l % 8 < 4 else (l // 8) + 1 for l in lengths]
    tasks = [veo_prompters[i].run(chunks[i], n_descriptions[i])
             for i in range(len(chunks))]
    result = await asyncio.gather(*tasks)

    descriptions = {}
    for i, r in enumerate(result):
        descriptions[chunks[i]] = r.model_dump()['descriptions']

    with open("descriptions.json", "w") as f:
        json.dump(descriptions, f)

asyncio.run(main())
