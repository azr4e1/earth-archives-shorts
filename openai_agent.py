from my_agents import Agent
from pydantic import BaseModel
from utils import read_prompt
import openai
import os
import logging
import asyncio


class OpenaiAgent(Agent):
    def __init__(self,
                 name: str,
                 model: str,
                 system_prompt: str,
                 api_key: str = None,
                 vector_store_id: int = None,
                 structured_text: BaseModel = None,
                 semaphore: asyncio.Semaphore = None,
                 settings: dict = None):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        client = openai.AsyncOpenAI(api_key=api_key)
        super().__init__(name, client, model)
        self.semaphore = semaphore
        self.system_prompt = system_prompt
        self.vector_store_id = vector_store_id
        self.structured_text = structured_text
        self.settings = settings if settings is not None else {}

        self.system_prompt += """\n\nYour input will be structured as a sequence of key-value pairs as follows:
### <KEY1> ###
<VALUE1>

### <KEY2> ###
<VALUE2>

...
"""

    async def run(self, query: str = None, **kwargs):
        if self.semaphore is None:
            result = await self._run(query, **kwargs)
            return result

        async with self.semaphore:
            result = await self._run(query, **kwargs)
            return result

    async def _run(self, query: str = None, **kwargs):
        self.log("started.", logging.INFO)

        inputs = kwargs.copy()
        if query is not None:
            inputs['knowledge'] = await self._rag(query)

        prompt = ""
        for k, v in inputs.items():
            if v is None:
                continue
            prompt += f"### {k} ###\n"
            prompt += f"{v}\n\n"

        prompt += "### answer ###"
        self.log("awaiting for response.")

        result = None
        if self.structured_text is not None:
            response = await self.client.responses.parse(
                model=self.model,
                input=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                text_format=self.structured_text,
                **self.settings
            )
            self.log("returned structured response.")
            result = response.output_parsed
        else:
            response = await self.client.responses.create(
                model=self.model,
                input=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                **self.settings
            )
            self.log("returned unstructured response.")
            result = response.output_text

        self.log("completed.", logging.INFO)
        return result

    async def _rag(self, query: str) -> str:
        if query is None:
            query = ""
        if self.vector_store_id is not None and query != "":
            self.log("querying vector store.")
            query_result = await self.client.vector_stores.search(
                vector_store_id=self.vector_store_id,
                query=query)
            context = []
            for data in query_result.model_dump()['data']:
                content = data['content']
                for el in content:
                    ctx = el.get('text', "")
                    context.append(ctx)
            self.log("finished querying vector store.")
            return "\n\n".join(context)
        elif query != "":
            raise Exception("Provide a valid vector store ID.")
        else:
            return None


class WriterAgent(OpenaiAgent):
    def __init__(self,
                 name: str,
                 model: str,
                 vector_store_id: int,
                 api_key: str = None,
                 settings: dict = None):
        structured_text = None
        super().__init__(name, model, read_prompt(self.__class__.__name__), api_key,
                         vector_store_id, structured_text, settings)

    async def run(self, query):
        result = await super().run(query=query, question=query)
        return result


class ChunkerSchema(BaseModel):
    descriptions: list[str]


class ChunkerAgent(OpenaiAgent):
    def __init__(self,
                 name: str,
                 model: str,
                 api_key: str = None,
                 settings: dict = None):
        vector_store_id = None
        structured_text = ChunkerSchema
        super().__init__(name, model, read_prompt(self.__class__.__name__), api_key,
                         vector_store_id, structured_text, settings)


class VeoPrompter(OpenaiAgent):
    def __init__(self,
                 name: str,
                 model: str,
                 api_key: str = None,
                 settings: dict = None):
        vector_store_id = None
        structured_text = ChunkerSchema
        super().__init__(name, model, read_prompt(self.__class__.__name__), api_key,
                         vector_store_id, structured_text, settings)

    async def run(self, script: str, versions: int, context: str = None) -> ChunkerSchema:
        result = await super().run(script=script, versions=versions, context=context)
        if context is None:
            return result
        result_list = result.model_dump()['descriptions']
        augmented_list = []
        for i in result_list:
            prompt = f"### CONTEXT ###\n{
                context}\n\n### VIDEO INSTRUCTIONS ###\n{i}"
            augmented_list.append(prompt)
        new_result = ChunkerSchema(descriptions=augmented_list)
        return new_result
