import time
import sys
from google import genai
from google.genai import types

client = genai.Client()

prompt = "Smooth tracking shot spiraling along a massive tree branch, canopy-level fantasy forest: The camera begins with a wide shot of towering, intertwining trees whose trunks glow faintly with bioluminescence, their branches forming natural platforms and walkways hundreds of feet above the ground. Living vines connect multiple platforms, swaying gently. The shot transitions to follow three Arboreal Volani—slender, graceful beings standing approximately six feet tall, their bodies covered in iridescent, leaf-like plumage that shifts from emerald to gold as they move. One Volani spreads their arms and glides gracefully from a high platform to a lower branch, plumage catching the light mid-flight. As the camera continues its spiral movement, two Volani on a central platform face each other, their hands moving in fluid, deliberate gestures—one raises their palm upward in a sweeping arc, the other responds by tracing a circular pattern in the air. A third Volani tilts their head back, opens their mouth, and begins singing—a visible concentration on their face. The other two join, heads bobbing slightly in rhythm, creating a visible sense of harmony. The camera completes its spiral, pulling back slightly to reveal five more Volani on neighboring platforms, all engaged in this synchronized musical communication. Soft, diffused lighting filters down through layers of leaves above, creating dappled green-gold illumination that shifts as branches sway. The overall atmosphere is tranquil and ancient. Audio: Layered harmonic singing with melodic, vowel-rich tones; gentle rustling of leaves and vine movement; distant bird-like calls; soft ambient forest hum."

operation = client.models.generate_videos(
    model="veo-3.0-fast-generate-001",
    prompt=prompt,
    config=types.GenerateVideosConfig(
        aspectRatio="9:16", durationSeconds="8")
)

# Poll the operation status until the video is ready.
while not operation.done:
    print("Waiting for video generation to complete...")
    time.sleep(10)
    operation = client.operations.get(operation)

# Download the generated video.
if operation.response.generated_videos is None:
    print("Could not generate video.")
    sys.exit(1)
generated_video = operation.response.generated_videos[0]
client.files.download(file=generated_video.video)
generated_video.video.save("volani.mp4")
print("Generated video saved to dialogue_example.mp4")
