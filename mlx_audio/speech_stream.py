"""Encode a speech response as one audio stream, without accessing MLX."""

import asyncio
import contextlib
import shutil
import struct
from dataclasses import dataclass

import numpy as np


@dataclass
class AudioChunk:
    audio: np.ndarray
    sample_rate: int


_CODECS = {
    "mp3": ("libmp3lame", "mp3"),
    "opus": ("libopus", "opus"),
    "ogg": ("libvorbis", "ogg"),
    "vorbis": ("libvorbis", "ogg"),
    "flac": ("flac", "flac"),
    "webm": ("libopus", "webm"),
}


def validate_audio_format(format):
    if format not in {"pcm", "wav", *_CODECS}:
        raise ValueError(f"Unsupported output format: {format}")
    if format in _CODECS and not shutil.which("ffmpeg"):
        raise ValueError(f"ffmpeg is required for {format} output")


def _mono(chunk):
    samples = np.asarray(chunk.audio, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or chunk.sample_rate <= 0:
        raise ValueError(
            "Expected mono or samples-by-channels audio and a positive sample rate"
        )
    return np.clip(np.nan_to_num(samples), -1, 1)


def _pcm16(samples):
    return np.clip(np.rint(samples * 32768), -32768, 32767).astype("<i2").tobytes()


def _wav_header(rate):
    # Streaming WAV has no known final length. Use the conventional unknown-size
    # marker once; all subsequent bytes are PCM, never additional WAV files.
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        rate,
        rate * 2,
        2,
        16,
        b"data",
        0xFFFFFFFF,
    )


async def encode_audio_stream(source, format):
    """Accept pre-encoded buffered output or worker-materialized audio chunks."""
    try:
        try:
            first = await anext(source)
        except StopAsyncIteration:
            return
        if isinstance(first, bytes):
            yield first
            async for chunk in source:
                yield chunk
            return

        rate = first.sample_rate

        async def chunks():
            yield first
            async for chunk in source:
                if chunk.sample_rate != rate:
                    raise ValueError("Sample rate changed during speech generation")
                yield chunk

        if format == "wav" or (format == "pcm" and rate == 24000):
            header = _wav_header(rate) if format == "wav" else b""
            async for chunk in chunks():
                data = _pcm16(_mono(chunk))
                if data:
                    yield header + data
                    header = b""
            return

        # A single process preserves codec and resampling state across chunks.
        async with contextlib.aclosing(
            _ffmpeg_stream(chunks(), rate, format)
        ) as encoder:
            async for data in encoder:
                yield data
    finally:
        await source.aclose()


async def _ffmpeg_stream(chunks, rate, format):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError("ffmpeg is required for encoding or resampling this response")
    codec, container = ("pcm_s16le", "s16le") if format == "pcm" else _CODECS[format]
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-probesize",
        "32",
        "-analyzeduration",
        "0",
        "-f",
        "f32le",
        "-ar",
        str(rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-ac",
        "1",
        "-c:a",
        codec,
    ]
    if format == "pcm":
        command += ["-ar", "24000"]
    if format in {"opus", "ogg", "vorbis"}:
        command += ["-page_duration", "20000"]
    command += ["-flush_packets", "1", "-f", container, "pipe:1"]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def feed():
        try:
            async for chunk in chunks:
                process.stdin.write(_mono(chunk).astype("<f4").tobytes())
                await process.stdin.drain()
        finally:
            process.stdin.close()

    async def errors():
        tail = b""
        while data := await process.stderr.read(4096):
            tail = (tail + data)[-16384:]
        return tail.decode(errors="replace")

    writer = asyncio.create_task(feed())
    stderr = asyncio.create_task(errors())
    try:
        while data := await process.stdout.read(8192):
            yield data
        await writer
        code = await process.wait()
        detail = await stderr
        if code:
            raise RuntimeError(f"ffmpeg speech encoding failed: {detail}")
    finally:
        writer.cancel()
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
        await asyncio.gather(writer, return_exceptions=True)
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
        await stderr
        await chunks.aclose()
