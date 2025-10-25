from openai_agent import WriterAgent, ChunkerAgent, VeoPrompter
from elevenlabs_agents import VoiceGenerationAgent
from video_generation import VideoGenerationAgent
from utils import Cacher, generate_hash
import asyncio
from pathlib import Path
from collections import defaultdict

ELEVENLABS_SEMAPHORE = 3
VEO_SEMAPHORE = 2
SAVE_DIR = Path("./.cache/20251023110303_safe")
OPENAI_MODEL = 'gpt-4.1'
VECTOR_STORE_ID = "vs_68f01ec9d8a08191b2ace026d2cf8a80"
VOICE_ID = "nrbjbLmJZ7T1FcsFbbeE"
ELEVENLABS_MODEL = "eleven_multilingual_v2"
VOICE_SPEED = 1.1
ASPECT_RATIO = "9:16"
VIDEO_DURATION = "8"


async def process_script(query):
    writer = WriterAgent('Writer', OPENAI_MODEL,
                         vector_store_id=VECTOR_STORE_ID)
    script = await writer.run(query=query)
    return script


async def process_chunks(script):
    chunker = ChunkerAgent('Chunker', OPENAI_MODEL)
    chunks_model = await chunker.run(script=script)
    chunks = chunks_model.model_dump()['descriptions']

    return chunks


async def process_voice(chunks, cacher=None):
    chunks_l = len(chunks)
    semaphore = asyncio.Semaphore(ELEVENLABS_SEMAPHORE)
    audio_agents = [VoiceGenerationAgent(f"AudioGeneration_{i}",
                                         voice_id=VOICE_ID,
                                         model=ELEVENLABS_MODEL,
                                         semaphore=semaphore,
                                         settings={'speed': VOICE_SPEED})
                    for i in range(chunks_l)]

    async def download_and_save_audio(agent, chunk):
        audio = await agent.run(chunk)
        audio_hash = generate_hash(chunk)
        if cacher:
            cacher.save_audio({audio_hash: audio})
        return audio_hash, audio

    results = await asyncio.gather(*[download_and_save_audio(audio_agents[i], chunks[i])
                                     for i in range(chunks_l)])
    audios = {audio_hash: audio for audio_hash, audio in results}

    return audios


async def process_veo_prompts(chunks, audios, context=None):
    chunks_l = len(chunks)
    veo_prompters = [VeoPrompter(
        f'Prompter_{i}', OPENAI_MODEL) for i in range(chunks_l)]
    lengths = [len(audios[generate_hash(chunk)]) / 1000 for chunk in chunks]
    n_descriptions = [int(l // 8) if l %
                      8 < 4 else int(l // 8) + 1 for l in lengths]
    result = await asyncio.gather(*[veo_prompters[i].run(chunks[i], n_descriptions[i], context)
                                    for i in range(chunks_l)])

    descriptions = {}
    for i, r in enumerate(result):
        descriptions[chunks[i]] = r.model_dump()['descriptions']

    return descriptions


async def process_video(descriptions, cacher=None):
    video_semaphore = asyncio.Semaphore(VEO_SEMAPHORE)
    descriptions_flat = []
    video_names = []
    for chunk, descs in descriptions.items():
        chunk_hash = generate_hash(chunk)
        for desc in descs:
            desc_hash = generate_hash(desc)
            video_names.append(f"{chunk_hash}_{desc_hash}")
            descriptions_flat.append(desc)

    video_agents = [VideoGenerationAgent(f"VideoGeneration_{i}",
                                         semaphore=video_semaphore,
                                         settings={"aspectRatio": ASPECT_RATIO,
                                                   "durationSeconds": VIDEO_DURATION})
                    for i in range(len(descriptions_flat))]

    async def download_and_save_video(agent, desc, video_name):
        video = await agent.run(desc)
        if cacher:
            cacher.save_videos({video_name: video})
        return video_name, video

    results = await asyncio.gather(*[download_and_save_video(video_agents[i], descriptions_flat[i], video_names[i])
                                     for i in range(len(descriptions_flat))])
    videos = {video_name: video for video_name, video in results}

    return videos


async def main():
    cacher = Cacher(save_dir=SAVE_DIR)
    script, chunks, audios, descriptions, videos = cacher.restore()
    query = 'Write a script about the Ortheans'

    if script is None:
        script = await process_script(query)
        cacher.save_script(script)

    if chunks is None:
        chunks = await process_chunks(script)
        cacher.save_chunks(chunks)

    if audios is None:
        audios = await process_voice(chunks, cacher)
    else:
        # calculate how many audios we are missing
        remaining_chunks = []
        for chunk in chunks:
            name = generate_hash(chunk)
            if name not in audios:
                remaining_chunks.append(chunk)
        if remaining_chunks:
            remaining_audios = await process_voice(remaining_chunks, cacher)
            audios.update(remaining_audios)

    if descriptions is None:
        context = None
        context_path = Path("./context.txt")
        if context_path.exists():
            context = context_path.read_text()
        descriptions = await process_veo_prompts(chunks, audios, context)
        cacher.save_descriptions(descriptions)

    # implement retrieval
    if videos is None:
        videos = await process_video(descriptions, cacher)
    else:
        # calculate how many audios we are missing
        remaining_descriptions = defaultdict(list)
        for chunk, descs in descriptions.items():
            chunk_hash = generate_hash(chunk)
            for desc in descs:
                desc_hash = generate_hash(desc)
                name = f"{chunk_hash}_{desc_hash}"
                if name not in videos:
                    remaining_descriptions[chunk].append(desc)
        if remaining_descriptions:
            remaining_videos = await process_video(remaining_descriptions, cacher)
            videos.update(remaining_videos)

asyncio.run(main())
