import os
from openai import OpenAI
client = OpenAI()

vector_store = client.vector_stores.create(
    name="The Earth Archives"
)

print(f"Vector Store Created: {vector_store.id}")

print("Loading files...")
client.vector_stores.file_batches.upload_and_poll(        # Upload file
    vector_store_id=vector_store.id,
    files=[open(f"./wiki/{f}", "rb") for f in os.listdir("./wiki")]
)
