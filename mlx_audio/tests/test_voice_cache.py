"""Persistence failures must rebuild prompts, never silently change a voice."""

import numpy as np
import pytest

from mlx_audio.voice_registry import VoiceRegistry, model_cache_identity


def prompt():
    return {
        "speaker_embedding": np.array([[0.25, 0.5]], dtype=np.float32),
        "ref_codes": np.zeros((1, 2, 3), dtype=np.int32),
        "ref_text_ids": np.array([[1, 2]], dtype=np.int32),
    }


@pytest.fixture
def registry(tmp_path):
    result = VoiceRegistry(tmp_path / "voices")
    result.create_voice("alice", "consent", b"original audio", transcript="Hello")
    result.store_prompt_cache("alice", "qwen", prompt(), model_identity="revision1")
    return result


def test_revision_transcript_and_recording_invalidate_cache(registry):
    def load():
        return registry.load_prompt_cache("alice", "qwen", model_identity="revision1")

    assert load() is not None
    assert (
        registry.load_prompt_cache("alice", "qwen", model_identity="revision2") is None
    )
    registry.update_voice("alice", transcript="Changed transcript")
    assert load() is None
    registry.update_voice("alice", transcript="Hello")
    assert load() is not None
    registry.get_audio_sample_path("alice").write_bytes(b"changed audio")
    assert load() is None


@pytest.mark.parametrize("damage", ["missing", "shape", "dtype", "nan", "truncated"])
def test_invalid_cache_is_a_miss(registry, damage):
    path = registry.get_prompt_cache_path("alice", "qwen")
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    if damage == "missing":
        del arrays["ref_codes"]
    elif damage == "shape":
        arrays["ref_codes"] = np.zeros((3, 2), dtype=np.int32)
    elif damage == "dtype":
        arrays["ref_text_ids"] = np.array([[1.5]])
    elif damage == "nan":
        arrays["speaker_embedding"][0, 0] = np.nan
    if damage == "truncated":
        path.write_bytes(path.read_bytes()[:64])
    else:
        np.savez(path, **arrays)
    assert (
        registry.load_prompt_cache("alice", "qwen", model_identity="revision1") is None
    )
    registry.store_prompt_cache("alice", "qwen", prompt(), model_identity="revision1")
    assert (
        registry.load_prompt_cache("alice", "qwen", model_identity="revision1")
        is not None
    )


def test_failed_cache_write_preserves_previous_cache(registry, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="disk full"):
        registry.store_prompt_cache(
            "alice", "qwen", prompt(), model_identity="revision2"
        )
    assert (
        registry.load_prompt_cache("alice", "qwen", model_identity="revision1")
        is not None
    )
    assert (
        len(list(registry.get_prompt_cache_path("alice", "qwen").parent.iterdir())) == 1
    )


def test_model_identity_tracks_tokenizer_config_and_weights(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    previous = model_cache_identity(tmp_path)
    assert model_cache_identity(tmp_path) == previous
    for path, contents in [
        (tokenizer, b"new tokenizer"),
        (config, b"new config"),
        (weights, b"new weights"),
    ]:
        path.write_bytes(contents)
        current = model_cache_identity(tmp_path)
        assert current != previous
        previous = current
