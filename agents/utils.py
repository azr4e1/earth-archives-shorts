from pathlib import Path
from datetime import datetime
import json
from pydub import AudioSegment
from hashlib import md5, algorithms_available


def read_prompt(name: str):
    with open("prompts.json") as f:
        prompts = json.load(f)

    prompt = prompts.get(name, None)
    return prompt


def generate_hash(string: str):
    return md5(string.encode()).hexdigest()


class Cacher:
    def __init__(self, basedir: str = ".", save_dir: Path = None):
        if save_dir is not None and isinstance(save_dir, Path) and save_dir.exists():
            self.save_dir = save_dir
            return

        cache = Path(basedir) / Path(".cache")
        cache.mkdir(exist_ok=True)

        save_dir = cache / \
            Path(f"{datetime.now().strftime('%Y%m%d%H%M%S')}_safe")
        save_dir.mkdir(exist_ok=True)

        self.save_dir = save_dir

    def save_script(self, script: str):
        with open(self.save_dir / "script.txt", "w") as f:
            f.write(script)

    def save_chunks(self, chunks: dict):
        with open(self.save_dir / "chunks.json", "w") as f:
            json.dump(chunks, f)

    def save_audio(self, audios: dict[str, AudioSegment]):
        audio_dir = self.save_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        for n, audio in audios.items():
            name = audio_dir / f"{n}.mp3"
            audio.export(str(name), format='mp3')

    def save_descriptions(self, descriptions: dict):
        with open(self.save_dir / "descriptions.json", "w") as f:
            json.dump(descriptions, f)

    def save_videos(self, videos: dict[str, bytearray]):
        video_dir = self.save_dir / "video"
        video_dir.mkdir(exist_ok=True)
        for n, video in videos.items():
            name = video_dir / f"{n}.mp4"
            with open(str(name), "wb") as f:
                f.write(video)

    def restore_script(self):
        script_path = self.save_dir / "script.txt"
        if not script_path.exists():
            return None
        script = script_path.read_text()
        return script

    def restore_chunks(self):
        chunks_path = self.save_dir / "chunks.json"
        if not chunks_path.exists():
            return None
        with open(chunks_path) as f:
            chunks = json.load(f)

        return chunks

    def restore_audio(self):
        audio_dir = self.save_dir / "audio"
        if not audio_dir.exists():
            return None
        audios = {}
        for f in audio_dir.iterdir():
            name = f.stem
            audio = AudioSegment.from_mp3(f)
            audios[name] = audio

        return audios

    def restore_descriptions(self):
        descriptions_path = self.save_dir / "descriptions.json"
        if not descriptions_path.exists():
            return None
        with open(descriptions_path) as f:
            descriptions = json.load(f)

        return descriptions

    def restore_videos(self):
        video_dir = self.save_dir / "video"
        if not video_dir.exists():
            return None
        videos = {}
        for f in video_dir.iterdir():
            name = f.stem
            video = f.read_bytes()
            videos[name] = video

        return videos

    def restore(self):
        script = self.restore_script()
        chunks = self.restore_chunks()
        audios = self.restore_audio()
        description = self.restore_descriptions()
        videos = self.restore_videos()

        return script, chunks, audios, description, videos
