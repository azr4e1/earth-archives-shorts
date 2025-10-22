from my_agents import Agent
from elevenlabs.client import AsyncElevenLabs
import os
import asyncio
import logging
from pydub import AudioSegment
import io


class VoiceGenerationAgent(Agent):
    def __init__(self,
                 name: str,
                 voice_id: str,
                 model: str,
                 api_key: str = None,
                 semaphore: asyncio.Semaphore = None,
                 settings: dict = None):
        if api_key is None:
            api_key = os.getenv("ELEVENLABS_API_KEY")
        client = AsyncElevenLabs(api_key=api_key)
        super().__init__(name, client, model)
        self.voice_id = voice_id
        self.settings = settings if settings is not None else {}
        self.semaphore = semaphore

    async def run(self, text: str) -> AudioSegment:
        if self.semaphore is None:
            result = await self._run(text,)
            return result

        async with self.semaphore:
            result = await self._run(text)
            return result

    async def _run(self, text: str):
        self.log("started.", logging.INFO)
        self.log("awaiting for response.")
        audio = self.client.text_to_speech.convert(
            text=text,
            voice_id=self.voice_id,
            model_id=self.model,
            output_format="mp3_44100_128",
            voice_settings=self.settings
        )

        buffer = bytearray()
        async for chunk in audio:
            buffer.extend(chunk)

        self.log("completed.", logging.INFO)

        audio = AudioSegment.from_mp3(io.BytesIO(buffer))
        return audio
