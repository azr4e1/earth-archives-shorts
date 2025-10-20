from my_agents import Agent
from pydantic import BaseModel
import openai
import os


class OpenaiAgent(Agent):
    def __init__(self,
                 name: str,
                 model: str,
                 system_prompt: str,
                 api_key: str = None,
                 vector_store_id: int = None,
                 structured_text: BaseModel = None,
                 settings: dict = None):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        client = openai.AsyncOpenAI(api_key=api_key)
        super().__init__(name, client, model)
        self.system_prompt = system_prompt
        self.vector_store_id = vector_store_id
        self.structured_text = structured_text
        self.settings = settings

        self.system_prompt += """\n\nYour input will be structured as a sequence of key-value pairs as follows:
### <KEY1> ###
<VALUE1>

### <KEY2> ###
<VALUE2>

...
"""

    async def run(self, query: str = None, **kwargs):
        inputs = kwargs.copy()
        if query is not None:
            inputs['knowledge'] = await self._rag(query)

        prompt = ""
        for k, v in inputs.items():
            prompt += f"### {k} ###\n"
            prompt += f"{v}\n\n"
        if self.structured_text is not None:
            response = await self.client.responses.parse(
                model=self.model,
                input=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                text_format=self.structured_text,
            )
            return response.output_parsed
        else:
            response = await self.client.responses.create(
                model=self.model,
                input=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
            )
            return response.output_text

    async def _rag(self, query: str) -> str:
        if query is None:
            query = ""
        if self.vector_store_id is not None and query != "":
            query_result = await self.client.vector_stores.search(
                vector_store_id=self.vector_store_id,
                query=query)
            context = []
            for data in query_result.model_dump()['data']:
                content = data['content']
                for el in content:
                    ctx = el.get('text', "")
                    context.append(ctx)
            return "\n\n".join(context)
        elif query != "":
            raise Exception("Provide a valid vector store ID.")
        else:
            return None
