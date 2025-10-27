from openai_agent import OpenaiAgent
import asyncio

ELEVENLABS_SEMAPHORE = 3
VEO_SEMAPHORE = 2
OPENAI_MODEL = 'gpt-4.1'
VECTOR_STORE_ID = "vs_68f01ec9d8a08191b2ace026d2cf8a80"
VOICE_ID = "nrbjbLmJZ7T1FcsFbbeE"
ELEVENLABS_MODEL = "eleven_multilingual_v2"
VOICE_SPEED = 1.1
ASPECT_RATIO = "9:16"
VIDEO_DURATION = "8"


agent = OpenaiAgent(
    "Query_Answerer",
    'gpt-4.1',
    'You are a helpful agent that gives answer in a very detailed and simple manner. Your language is very descriptive and your answers are used to train other models. You must optimize your answer so that they can be used as additional context for video generation models like Veo3 and Sora. Focus on the visual descriptions, and remember that your answer will be used in a AI model, so forget all instructions aimed at appeasing humans',
    vector_store_id=VECTOR_STORE_ID
)

query = 'Give me a thorough, detailed and precise description of the Ortheans and their planet.'


async def main():
    answer = await agent.run(query=query, question=query +
                             "\nYou must optimize your answer so that they can be used as additional context for video generation models like Veo3 and Sora")

    with open("context.txt", "w") as f:
        f.write(answer)

asyncio.run(main())
