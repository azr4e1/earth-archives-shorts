from openai_agent import OpenaiAgent
import openai
import asyncio

query = 'What are the intelligent species of the Helionis Cluster?'
agent = OpenaiAgent('test', openai.OpenAI(), 'gpt-4.1',
                    'You are a helpful assistant',
                    vector_store_id="vs_68f01ec9d8a08191b2ace026d2cf8a80")

(asyncio.run(agent.run(query=query, question=query)))
