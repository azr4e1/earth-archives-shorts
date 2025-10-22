from my_agents import Agent
from google.genai import types
from google import genai
import os
import asyncio
from io import BytesIO
import logging


class VideoGenerationAgent(Agent):
    def __init__(self,
                 name: str,
                 api_key: str = None,
                 semaphore: asyncio.Semaphore = None,
                 settings: dict = None):

        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key)
        model_names = ["veo-3.0-generate-001",
                       "veo-3.0-fast-generate-001", "veo-2.0-generate-001"]
        super().__init__(name, client, model_names)
        self.semaphore = semaphore
        self.settings = settings if settings is not None else {}

    async def run(self, prompt: str):
        if self.semaphore is None:
            result = await self._run(prompt,)
            return result

        async with self.semaphore:
            result = await self._run(prompt)
            return result

    async def _run(self, prompt: str):
        self.log("started.", logging.INFO)
        for model in self.model:
            try:
                # Use async version of generate_videos
                self.log(f"trying model {model}")
                operation = await self.client.aio.models.generate_videos(
                    model=model,
                    prompt=prompt,
                    config=types.GenerateVideosConfig(**self.settings)
                )

                # Poll asynchronously
                while not operation.done:
                    await asyncio.sleep(10)  # Non-blocking sleep
                    operation = await self.client.aio.operations.get(operation)

                generated_video = operation.response.generated_videos[0]

                # Download asynchronously - populates video_bytes
                video_buffer = await self.client.aio.files.download(file=generated_video.video)

                # # Access the bytes
                # video_buffer = BytesIO(generated_video.video.video_bytes)

                self.log("completed.", logging.INFO)
                return video_buffer

            except Exception as e:
                self.log(f"model {model} failed: {e}")
                continue

        raise Exception("couldn't generate video.")
