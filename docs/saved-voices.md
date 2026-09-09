# Saved voices and streamed speech

Start the server with `mlx_audio.server --host 127.0.0.1 --port 8000`.
Install FFmpeg (`brew install ffmpeg` on macOS) for MP3/Opus encoding and
PCM resampling from models whose output rate is not 24 kHz.

## Save and reuse a voice

Upload a reference recording once. Include its transcript for Qwen3 Base
transcript-guided cloning; omit it to use the speaker embedding alone.

```bash
curl http://127.0.0.1:8000/v1/audio/voices \
  -F 'name=my-voice' \
  -F 'consent=local-reference' \
  -F 'audio_sample=@reference.wav' \
  -F 'transcript=The exact words spoken in reference.wav.'
```

The response includes `id`, `name`, `object`, and `created_at`. The `consent`
field is stored locally as metadata; this server does not create or validate
OpenAI voice-consent records.

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit","input":"Hello from my saved voice.","voice":{"id":"my-voice"},"lang_code":"English","response_format":"mp3"}' \
  --output speech.mp3
```

`"voice":"my-voice"` also works. Unmatched string names retain model-specific
built-in voice behavior; an unknown explicit `{"id":"…"}` returns 404.
Saved voices cannot be combined with `ref_audio` or `ref_text` in a speech request.

- `GET /v1/audio/voices` returns `{"voices":["my-voice"]}`.
- `GET /v1/audio/voices?model=…` retains upstream's model voice-pack listing.
- `DELETE /v1/audio/voices/my-voice` removes the recording and cached prompts.

Recordings and Qwen3 prompts persist under `~/.cache/mlx-audio/voices`, configurable
with `MLX_AUDIO_VOICES_DIR`. Existing saved voices remain usable. Version-1 prompt
caches are rebuilt on next use with the corrected Qwen tokenizer; recordings
and metadata are preserved. Qwen3 Base prompts are computed on first use and reused after
server restarts. Other reference-audio models receive the saved recording and
transcript. Saved-voice requests run serially on the inference worker; ordinary
requests retain upstream batching.

## Streaming

Set `stream: true` to enable incremental model generation. The default remains
`false`, which returns a complete encoded recording. `streaming_interval` controls
model chunk duration (default 2 seconds); actual latency also depends on model
loading, first-use prompt preparation, and codec buffering.

```bash
curl --no-buffer http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit","input":"This audio arrives while speech is still being generated.","voice":"my-voice","lang_code":"English","response_format":"pcm","stream":true,"streaming_interval":0.32}' \
  | ffplay -autoexit -f s16le -ar 24000 -ac 1 -i pipe:0
```

PCM responses are headerless signed 16-bit little-endian mono at 24 kHz, including
buffered PCM responses. WAV streams have one header with unknown-length markers
and use the model's sample rate; applications requiring finalized WAV sizes
should use `stream: false`. MP3 and Opus streams use one encoder for the whole
response, with output drained during generation and finalized on completion.
The upstream FLAC, Ogg/Vorbis, and WebM formats are retained too. Responses contain
binary audio, not SSE events. Failures before the first encoded chunk return an
HTTP error; failures after headers terminate the stream. Disconnecting cancels
inference and closes the encoder.

## OpenClaw

Configure the OpenAI TTS provider to use your local server. This example uses the
current `tts.providers.openai` configuration shape:

```json5
{
  tts: {
    provider: "openai",
    providers: {
      openai: {
        baseUrl: "http://127.0.0.1:8000/v1",
        apiKey: "local",
        model: "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
        voice: "my-voice",
        extraBody: { lang_code: "English" }
      }
    }
  }
}
```

Use an address reachable from the OpenClaw gateway; loopback refers to the
gateway's own host. The local API does not require a key, but OpenClaw requires
a nonempty provider key. A custom base URL permits local model and voice names.
Leaving `responseFormat` unset lets OpenClaw choose MP3 for files and Opus for
voice notes. Its telephony path requests 24 kHz PCM.

To request incremental server generation, add `stream: true` and optionally
`streaming_interval: 0.32` to `extraBody`. OpenClaw's current OpenAI provider
buffers the response before returning it, so this does not enable earlier
playback inside OpenClaw. No OpenClaw changes are required for compatibility.

References: [OpenClaw TTS configuration](https://docs.openclaw.ai/tools/tts),
[OpenClaw OpenAI speech provider](https://github.com/openclaw/openclaw/blob/main/extensions/openai/speech-provider.ts).
