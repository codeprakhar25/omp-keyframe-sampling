# Train pack (expressive Hindi TTS)

- rows: 247
- missing wav: 0
- tag1: {'thinking': 59, 'none': 31, 'inhaling': 27, 'sigh': 23, 'pause': 21, 'whispering': 16, 'sad': 14, 'excited': 13, 'angry': 12, 'fear': 11, 'surprise': 10, 'disgust': 10}

Fields in `manifest.jsonl`:
- `caption` — tagged text for conditioning (`[angry] ...`)
- `text` — plain transcript (no tags)
- `audio` — relative path under this folder

**Not a Fireworks SFT dataset.** Fireworks managed fine-tune is chat-LLM only.
Use this pack on a TTS trainer (RunPod / local / Modal).
