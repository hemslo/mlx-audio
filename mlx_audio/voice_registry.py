import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import mlx.core as mx
import numpy as np

VOICE_SAMPLE_MAX_BYTES = 10 * 1024 * 1024
# Version 2 rebuilds reference text IDs with Qwen's corrected tokenizer regex.
# Uploaded recordings and voice metadata are unaffected.
VOICE_CACHE_SCHEMA_VERSION = 2
SUPPORTED_VOICE_SAMPLE_MIME_TYPES = {
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/ogg": ".ogg",
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/webm": ".webm",
    "audio/mp4": ".mp4",
}


class VoiceRegistry:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "VoiceRegistry":
        default_root = Path.home() / ".cache" / "mlx-audio" / "voices"
        return cls(Path(os.getenv("MLX_AUDIO_VOICES_DIR", default_root)))

    def create_voice(
        self,
        name: str,
        consent: str,
        audio_bytes: bytes,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        transcript: Optional[str] = None,
    ) -> Dict[str, Any]:
        voice_id = self._normalize_voice_id(name)
        if self.resolve_voice(name) is not None:
            raise FileExistsError(
                f"Voice '{voice_id}' already exists. Choose a different name."
            )

        voice_dir = self._voice_dir(voice_id)
        voice_dir.mkdir(parents=True, exist_ok=True)

        suffix = self._detect_suffix(filename=filename, content_type=content_type)
        sample_name = f"audio_sample{suffix}"
        sample_path = voice_dir / sample_name
        sample_path.write_bytes(audio_bytes)

        metadata = {
            "id": voice_id,
            "object": "audio.voice",
            "created_at": int(time.time()),
            "name": name,
            "consent": consent,
            "audio_sample_filename": filename or sample_name,
            "audio_sample_mime_type": content_type,
            "audio_sample_path": sample_name,
            "transcript": transcript,
            "transcript_model": None,
        }
        self._write_metadata(voice_id, metadata)
        return self.public_voice(metadata)

    def has_voice(self, voice_id: str) -> bool:
        return self._metadata_path(voice_id).exists()

    def get_voice(self, voice_id: str) -> Dict[str, Any]:
        metadata_path = self._metadata_path(voice_id)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Voice '{voice_id}' not found")

        with metadata_path.open(encoding="utf-8") as f:
            return json.load(f)

    def resolve_voice(self, reference: str) -> Optional[Dict[str, Any]]:
        normalized_reference = self._normalize_voice_id(reference)
        if self.has_voice(normalized_reference):
            return self.get_voice(normalized_reference)

        for metadata_path in self.root.glob("*/metadata.json"):
            with metadata_path.open(encoding="utf-8") as f:
                metadata = json.load(f)

            name = metadata.get("name")
            if isinstance(name, str) and name.strip() == normalized_reference:
                return metadata

        return None

    def update_voice(self, voice_id: str, **fields: Any) -> Dict[str, Any]:
        metadata = self.get_voice(voice_id)
        metadata.update(fields)
        self._write_metadata(voice_id, metadata)
        return metadata

    def list_voices(self) -> list[Dict[str, Any]]:
        voices = []
        if not self.root.exists():
            return voices

        for metadata_path in self.root.glob("*/metadata.json"):
            with metadata_path.open(encoding="utf-8") as f:
                metadata = json.load(f)
            voices.append(self.public_voice(metadata))

        voices.sort(key=lambda voice: voice["name"])
        return voices

    def list_voice_names(self) -> list[str]:
        return [voice["name"] for voice in self.list_voices()]

    def get_audio_sample_path(self, voice_id: str) -> Path:
        metadata = self.get_voice(voice_id)
        return self._voice_dir(voice_id) / metadata["audio_sample_path"]

    def get_prompt_cache_path(self, voice_id: str, model_name: str) -> Path:
        model_hash = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:16]
        cache_dir = self._voice_dir(voice_id) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{model_hash}.npz"

    def load_prompt_cache(
        self, voice_id: str, model_name: str
    ) -> Optional[Dict[str, mx.array]]:
        cache_path = self.get_prompt_cache_path(voice_id, model_name)
        if not cache_path.exists():
            return None

        try:
            with np.load(cache_path, allow_pickle=False) as data:
                schema_version = int(np.array(data["schema_version"]).reshape(-1)[0])
                if schema_version != VOICE_CACHE_SCHEMA_VERSION:
                    return None

                prompt = {}
                if "ref_codes" in data.files:
                    prompt["ref_codes"] = mx.array(data["ref_codes"])
                if "ref_text_ids" in data.files:
                    prompt["ref_text_ids"] = mx.array(data["ref_text_ids"])
                if "speaker_embedding" in data.files:
                    prompt["speaker_embedding"] = mx.array(data["speaker_embedding"])
                return prompt or None
        except (KeyError, OSError, ValueError):
            return None

    def store_prompt_cache(
        self,
        voice_id: str,
        model_name: str,
        prompt: Dict[str, mx.array],
    ) -> Path:
        cache_path = self.get_prompt_cache_path(voice_id, model_name)
        arrays = {
            "schema_version": np.array([VOICE_CACHE_SCHEMA_VERSION], dtype=np.int32)
        }
        if "ref_codes" in prompt and prompt["ref_codes"] is not None:
            arrays["ref_codes"] = np.array(prompt["ref_codes"])
        if "ref_text_ids" in prompt and prompt["ref_text_ids"] is not None:
            arrays["ref_text_ids"] = np.array(prompt["ref_text_ids"])
        if "speaker_embedding" in prompt and prompt["speaker_embedding"] is not None:
            arrays["speaker_embedding"] = np.array(prompt["speaker_embedding"])

        np.savez_compressed(cache_path, **arrays)
        return cache_path

    def delete_voice(self, reference: str) -> Dict[str, Any]:
        metadata = self.resolve_voice(reference)
        if metadata is None:
            raise FileNotFoundError(f"Voice '{reference}' not found")

        voice_dir = self._voice_dir(metadata["id"])
        if voice_dir.exists():
            shutil.rmtree(voice_dir)
        return metadata

    @staticmethod
    def public_voice(metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": metadata["id"],
            "created_at": metadata["created_at"],
            "name": metadata["name"],
            "object": "audio.voice",
        }

    def _voice_dir(self, voice_id: str) -> Path:
        return self.root / voice_id

    def _metadata_path(self, voice_id: str) -> Path:
        return self._voice_dir(voice_id) / "metadata.json"

    def _write_metadata(self, voice_id: str, metadata: Dict[str, Any]) -> None:
        voice_dir = self._voice_dir(voice_id)
        voice_dir.mkdir(parents=True, exist_ok=True)
        with self._metadata_path(voice_id).open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    @staticmethod
    def _normalize_voice_id(name: str) -> str:
        voice_id = name.strip()
        if not voice_id:
            raise ValueError("name cannot be empty")
        if voice_id in {".", ".."}:
            raise ValueError("name cannot be '.' or '..'")
        if "/" in voice_id or "\\" in voice_id or "\x00" in voice_id:
            raise ValueError("name cannot contain path separators or null bytes")
        return voice_id

    @staticmethod
    def _detect_suffix(
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> str:
        if filename:
            suffix = Path(filename).suffix
            if suffix:
                return suffix
        if content_type:
            normalized = content_type.split(";", 1)[0].strip().lower()
            if normalized in SUPPORTED_VOICE_SAMPLE_MIME_TYPES:
                return SUPPORTED_VOICE_SAMPLE_MIME_TYPES[normalized]
        return ".wav"
