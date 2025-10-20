import time
import sys
from google import genai
from google.genai import types
import json
from tqdm import tqdm
import os


def generate_video(client, prompt, filename):
    model_names = ["veo-3.0-generate-001",
                   "veo-3.0-fast-generate-001", "veo-2.0-generate-001"]
    for model in model_names:
        try:
            operation = client.models.generate_videos(
                model=model,
                prompt=prompt,
                config=types.GenerateVideosConfig(
                    aspectRatio="9:16", durationSeconds="8")
            )

            # Poll the operation status until the video is ready.
            while not operation.done:
                operation = client.operations.get(operation)

            generated_video = operation.response.generated_videos[0]
            client.files.download(file=generated_video.video)
            generated_video.video.save(filename)
            return "Success"
        except Exception as e:
            print(e)
            continue

    return "Failure"


with open("../descriptions.json") as f:
    descriptions = json.load(f)


client = genai.Client()
descs = [d for _, descs in descriptions.items() for d in descs]
i = 0
print(descs[20:])
# generate_video(client, descs[0], "test.mp4")
# model = "veo-3.0-generate-001"
# operation = client.models.generate_videos(
#     model=model,
#     prompt=descs[0],
#     config=types.GenerateVideosConfig(
#         aspectRatio="9:16", durationSeconds="8")
# )
# while not operation.done:
#     operation = client.operations.get(operation)

# generated_video = operation.response.generated_videos[0]
# client.files.download(file=generated_video.video)
# generated_video.video.save("test.mp4")

# for d in (pbar := tqdm(descs)):
#     name = f"video_{i}.mp4"
#     status = generate_video(client, d, name)
#     pbar.display(status, pos=0)
#     i += 1
