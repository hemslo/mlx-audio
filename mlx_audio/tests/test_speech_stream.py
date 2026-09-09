import asyncio
import io
import shutil
import socket
import subprocess
import threading
import time
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient

import mlx_audio.server as server
from mlx_audio.audio_io import read as audio_read
from mlx_audio.speech_stream import AudioChunk, encode_audio_stream
from mlx_audio.voice_registry import VoiceRegistry


def test_old_prompt_cache_is_rebuilt_without_removing_voice(registry):
    import mlx.core as mx

    registry.create_voice("old-cache", "consent", b"recording", transcript="Hello")
    path = registry.get_prompt_cache_path("old-cache", "qwen")
    np.savez(path, schema_version=np.array([1]), ref_text_ids=np.array([[1, 2]]))
    assert registry.load_prompt_cache("old-cache", "qwen") is None
    assert registry.get_audio_sample_path("old-cache").read_bytes() == b"recording"
    assert registry.get_voice("old-cache")["transcript"] == "Hello"
    registry.store_prompt_cache(
        "old-cache", "qwen", {"ref_text_ids": mx.array([[3, 4]])}
    )
    assert registry.load_prompt_cache("old-cache", "qwen")["ref_text_ids"].tolist() == [
        [3, 4]
    ]


@pytest.fixture
def registry(tmp_path, monkeypatch):
    registry = VoiceRegistry(tmp_path / "voices")
    monkeypatch.setattr(server, "voice_registry", registry)
    return registry


@pytest.fixture
def provider(monkeypatch):
    class Model:
        def generate(self, text, **kwargs):
            for index in range(4):
                yield SimpleNamespace(
                    audio=np.sin(np.arange(12000) * 0.1).astype(np.float32) * 0.2,
                    sample_rate=24000,
                )

    model = Model()
    monkeypatch.setattr(server.model_provider, "load_model", lambda name: model)
    return model


@pytest.mark.parametrize("stream", [False, True])
def test_invalid_saved_voice_and_format_fail_before_headers(registry, provider, stream):
    with TestClient(server.app) as client:
        for fields, status in [
            ({"voice": {"id": "missing"}}, 404),
            ({"voice": {"id": "../escape"}}, 400),
            ({"response_format": "invalid"}, 400),
        ]:
            response = client.post(
                "/v1/audio/speech",
                json={
                    "model": "fake",
                    "input": "hello",
                    "response_format": "wav",
                    "stream": stream,
                    **fields,
                },
            )
            assert response.status_code == status
            assert "detail" in response.json()


def test_saved_voice_conflict_and_batch_exclusion(registry, provider):
    registry.create_voice("saved", "consent", b"sample")
    payload = server.SpeechRequest(model="fake", input="hello", voice="saved")
    request = server.InferenceRequest(
        endpoint_kind="tts",
        model_name="fake",
        payload=server.SpeechTaskPayload(payload),
    )
    provider.supports_tts_batch = lambda **kwargs: True
    provider.supports_tts_continuous_batch = lambda **kwargs: True
    provider.batch_generate = lambda texts, voices: iter(())
    provider.create_tts_batch_session = lambda options: None
    adapter = server.TTSExecutionAdapter()
    assert not adapter.supports_batch(request)
    assert not adapter.supports_continuous_batch(request)
    payload.voice = "builtin"
    assert adapter.supports_batch(request)
    assert adapter.supports_continuous_batch(request)
    with TestClient(server.app) as client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "fake",
                "input": "hello",
                "voice": "saved",
                "ref_text": "conflict",
            },
        )
        assert response.status_code == 400


@pytest.mark.parametrize("transcript", [None, "A reference transcript."])
def test_saved_prompt_survives_registry_restart(
    registry, provider, monkeypatch, transcript
):
    import mlx.core as mx

    created = []
    generated = []
    provider.model_type = "qwen3_tts"
    provider.config = SimpleNamespace(tts_model_type="base")

    def create_voice_clone_prompt(ref_audio, ref_text):
        assert threading.current_thread() is not threading.main_thread()
        created.append((ref_audio, ref_text))
        prompt = {"speaker_embedding": mx.array([[len(created)]], dtype=mx.float32)}
        if ref_text:
            prompt.update(
                ref_codes=mx.zeros((1, 2, 3), dtype=mx.int32),
                ref_text_ids=mx.array([[1, 2]], dtype=mx.int32),
            )
        return prompt

    provider.create_voice_clone_prompt = create_voice_clone_prompt
    original = provider.generate

    def generate(text, **kwargs):
        generated.append(kwargs)
        return original(text, **kwargs)

    provider.generate = generate
    for name in ["first", "second"]:
        registry.create_voice(name, "consent", b"sample", transcript=transcript)
    with TestClient(server.app) as client:
        for name in ["first", "first", "second"]:
            monkeypatch.setattr(server, "voice_registry", VoiceRegistry(registry.root))
            response = client.post(
                "/v1/audio/speech",
                json={
                    "model": "fake",
                    "input": "hello",
                    "voice": {"id": name},
                    "stream": True,
                    "response_format": "pcm",
                },
            )
            assert response.status_code == 200
    assert len(created) == 2
    assert created[0][1] == transcript
    assert all(
        item["voice"] is None and item["ref_audio"] is None for item in generated
    )
    assert [
        float(item["voice_clone_prompt"]["speaker_embedding"].item())
        for item in generated
    ] == [1, 1, 2]


def test_saved_voice_falls_back_without_passing_prompt_kwarg(
    registry, provider, monkeypatch
):
    from mlx_audio.audio_io import write

    buffer = io.BytesIO()
    write(buffer, np.zeros(2400, dtype=np.float32), 24000, format="wav")
    registry.create_voice(
        "fallback", "consent", buffer.getvalue(), filename="sample.wav"
    )
    provider.sample_rate = 24000
    original = provider.generate
    calls = []

    def generate(text, **kwargs):
        calls.append(kwargs)
        assert "voice_clone_prompt" not in kwargs
        assert kwargs["voice"] is None
        assert kwargs["ref_audio"] is not None
        return original(text, **kwargs)

    provider.generate = generate
    with TestClient(server.app) as client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "fake",
                "input": "hello",
                "voice": "fallback",
                "response_format": "wav",
            },
        )
        assert response.status_code == 200
    assert len(calls) == 1


def test_model_voice_listing_retains_upstream_shape(registry, monkeypatch, tmp_path):
    voice_dir = tmp_path / "snapshot" / "voices"
    voice_dir.mkdir(parents=True)
    (voice_dir / "af_heart.safetensors").touch()
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda **kwargs: str(voice_dir.parent)
    )
    with TestClient(server.app) as client:
        response = client.get("/v1/audio/voices?model=kokoro")
        assert response.status_code == 200
        assert response.json()["object"] == "list"
        assert response.json()["model"] == "kokoro"
        assert response.json()["data"][0]["id"] == "af_heart"


def test_encoder_preflight_and_inference_error(provider, monkeypatch):
    with TestClient(server.app) as client:
        monkeypatch.setattr("mlx_audio.speech_stream.shutil.which", lambda name: None)
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "fake",
                "input": "hello",
                "stream": True,
                "response_format": "mp3",
            },
        )
        assert response.status_code == 400
        assert "ffmpeg" in response.json()["detail"]

        def fail(*args, **kwargs):
            raise ValueError("bad voice prompt")

        provider.generate = fail
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "fake",
                "input": "hello",
                "stream": True,
                "response_format": "wav",
            },
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "bad voice prompt"


@pytest.mark.parametrize("stream", [False, True])
def test_empty_generation_is_400(provider, stream):
    provider.generate = lambda *args, **kwargs: iter(())
    with TestClient(server.app) as client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "fake",
                "input": "hello",
                "stream": stream,
                "response_format": "wav",
            },
        )
        assert response.status_code == 400


@pytest.mark.asyncio
async def test_pcm_continuous_resampling_matches_single_input():
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    audio = np.sin(np.arange(44100) * 0.03).astype(np.float32) * 0.5

    async def encode(parts):
        async def source():
            for part in parts:
                yield AudioChunk(part, 44100)

        return b"".join([chunk async for chunk in encode_audio_stream(source(), "pcm")])

    fragmented = await encode(np.array_split(audio, 37))
    whole = await encode([audio])
    assert fragmented == whole
    assert len(whole) == 24000 * 2


@pytest.mark.asyncio
async def test_pcm_clips_and_downmixes():
    async def source():
        yield AudioChunk(np.array([[-2, -2], [0, 0], [2, 2]], dtype=np.float32), 24000)

    data = b"".join([part async for part in encode_audio_stream(source(), "pcm")])
    assert np.frombuffer(data, dtype="<i2").tolist() == [-32768, 0, 32767]


@pytest.fixture
def http_server(provider):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    instance = uvicorn.Server(
        uvicorn.Config(server.app, log_level="error", lifespan="on")
    )
    thread = threading.Thread(
        target=instance.run, kwargs={"sockets": [sock]}, daemon=True
    )
    thread.start()
    deadline = time.monotonic() + 10
    while not instance.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert instance.started
    yield f"http://127.0.0.1:{port}"
    instance.should_exit = True
    thread.join(timeout=10)
    sock.close()
    assert not thread.is_alive()


@pytest.mark.parametrize("format", ["pcm", "wav", "mp3", "opus"])
def test_real_http_delivers_audio_before_generation_finishes(
    http_server, provider, format
):
    if format in {"mp3", "opus"} and not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    release = threading.Event()
    waiting = threading.Event()
    finished = threading.Event()

    def generate(*args, **kwargs):
        assert kwargs["stream"] is True
        # Enough input for each codec to produce audio, then block generation.
        for _ in range(8):
            yield SimpleNamespace(
                audio=np.sin(np.arange(12000) * 0.1).astype(np.float32) * 0.2,
                sample_rate=24000,
            )
        waiting.set()
        assert release.wait(10)
        finished.set()

    provider.generate = generate
    try:
        with httpx.stream(
            "POST",
            http_server + "/v1/audio/speech",
            json={
                "model": "fake",
                "input": "hello",
                "voice": "saved-name",
                "stream": True,
                "response_format": format,
            },
            timeout=10,
        ) as response:
            assert response.status_code == 200
            parts = response.iter_bytes()
            first = next(parts)
            while len(first) < 4096:
                first += next(parts)
            assert first
            assert not finished.is_set()
            release.set()
            data = first + b"".join(parts)
    finally:
        release.set()
    if format == "pcm":
        assert len(data) == 96000 * 2
    elif format == "wav":
        assert data.count(b"RIFF") == 1
        decoded, rate = audio_read(io.BytesIO(data))
        assert len(decoded) == 96000
        assert rate == 24000
    else:
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                "pipe:0",
                "-ar",
                "24000",
                "-f",
                "s16le",
                "pipe:1",
            ],
            input=data,
            capture_output=True,
            check=True,
        )
        assert abs(len(result.stdout) // 2 - 96000) < 2400


@pytest.mark.asyncio
async def test_encoder_close_terminates_process(monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    processes = []
    spawn = asyncio.create_subprocess_exec

    async def capture(*args, **kwargs):
        process = await spawn(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    closed = asyncio.Event()

    async def source():
        try:
            yield AudioChunk(np.zeros(96000, dtype=np.float32), 24000)
            await asyncio.Event().wait()
        finally:
            closed.set()

    encoded = encode_audio_stream(source(), "mp3")
    assert await asyncio.wait_for(anext(encoded), timeout=10)
    await asyncio.wait_for(encoded.aclose(), timeout=10)
    assert closed.is_set()
    assert processes[0].returncode is not None


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_http_disconnect_cancels_inference(http_server, provider, monkeypatch):
    handles = []
    broker = server.get_inference_broker()
    submit = broker.submit

    def capture(**kwargs):
        handle = submit(**kwargs)
        if kwargs["endpoint_kind"] == "tts":
            handles.append(handle)
        return handle

    monkeypatch.setattr(broker, "submit", capture)
    stopped = threading.Event()

    def generate(*args, **kwargs):
        try:
            for _ in range(500):
                yield SimpleNamespace(
                    audio=np.zeros(12000, dtype=np.float32), sample_rate=24000
                )
                time.sleep(0.01)
        finally:
            stopped.set()

    provider.generate = generate
    with httpx.stream(
        "POST",
        http_server + "/v1/audio/speech",
        json={
            "model": "fake",
            "input": "hello",
            "stream": True,
            "response_format": "mp3",
        },
        timeout=10,
    ) as response:
        assert response.status_code == 200
        assert next(response.iter_bytes())
    assert stopped.wait(5)
    assert handles[0].cancel_event.is_set()


@pytest.mark.asyncio
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
async def test_midstream_inference_error_is_not_silently_swallowed():
    async def source():
        yield AudioChunk(np.zeros(96000, dtype=np.float32), 24000)
        raise ValueError("generation failed")

    with pytest.raises(ValueError, match="generation failed"):
        async for _ in encode_audio_stream(source(), "mp3"):
            pass
