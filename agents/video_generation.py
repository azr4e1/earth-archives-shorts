from my_agents import Agent
from google.genai import types
from google import genai
import os
import asyncio


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
        for model in self.model:
            try:
                operation = self.client.models.generate_videos(
                    model=model,
                    prompt=prompt,
                    config=types.GenerateVideosConfig(**self.settings)
                )

                # Poll the operation status until the video is ready.
                while not operation.done:
                    operation = self.client.operations.get(operation)

                generated_video = operation.response.generated_videos[0]
                self.client.files.download(file=generated_video.video)
                generated_video.video.save(f"{self.name}.mp4")
                return "Success"
            except Exception:
                continue

        return "Failure"
