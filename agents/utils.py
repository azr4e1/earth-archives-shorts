from pathlib import Path
from datetime import datetime
import json
from pydub import AudioSegment


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

    def save_audio(self, audios: list[AudioSegment]):
        audio_dir = self.save_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        for i, audio in enumerate(audios):
            name = audio_dir / f"audio_{i}.mp3"
            audio.export(str(name), format='mp3')

    def save_descriptions(self, descriptions: dict):
        with open(self.save_dir / "descriptions.json", "w") as f:
            json.dump(descriptions, f)

    def save_videos(self, videos: list[bytearray]):
        video_dir = self.save_dir / "video"
        video_dir.mkdir(exist_ok=True)
        for i, video in enumerate(videos):
            name = video_dir / f"video_{i}.mp4"
            with open(str(name), "wb") as f:
                f.write(video)

    def restore_script(self):
        pass

    def restore_chunks(self):
        pass

    def restore_audio(self):
        pass

    def restore_descriptions(self):
        pass

    def restore_videos(self):
        pass

    def restore(self):
        script = self.restore_script()
        chunks = self.restore_chunks()
        audios = self.restore_audio()
        description = self.restore_descriptions()
        videos = self.restore_videos()

        return script, chunks, audios, description, videos
