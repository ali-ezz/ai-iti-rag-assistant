# Demo recording guide (silent video + live demo)

## A. Start the app (Docker)

```bash
cd ai-iti-rag-assistant
docker compose up -d --build
# wait ~1 min, then:
curl -s http://127.0.0.1:8000/health
# expect: {"status":"ok","vector_store":"ready","ollama":"ready",...}
```

Open:

- Chat UI: http://localhost:8501
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

First Ollama request downloads/pulls the model if the volume is cold (`qwen2.5:1.5b`).

## B. What to record (2–4 minutes, NO microphone)

Use `Cmd+Shift+5` → Record Selected Portion (or window). Narrate with on-screen actions only.

| Time | On screen |
|------|-----------|
| 0:00 | Browser opens http://localhost:8501 — show title + “How this assistant works” |
| 0:20 | Type: `How does Non-Max Suppression remove duplicate detections?` |
| 0:35 | Spinner → answer with `[Source …]` |
| 0:50 | Expand **Sources** — show `LAB-05-Object-Detection — … — chunk N` |
| 1:10 | Type: `What is the difference between stemming and lemmatization?` |
| 1:30 | Expand Sources → LAB-08 |
| 1:50 | New tab: http://localhost:8000/health → status ok |
| 2:10 | Optional: http://localhost:8000/docs → Execute GET /health and POST /query |
| 2:40 | New tab: https://github.com/ali-ezz/ai-iti-rag-assistant (brief scroll) |
| 3:00 | Stop recording |

Stop with the floating toolbar → file on Desktop as `.mov`.

### Convert to MP4 (optional)

```bash
ffmpeg -i ~/Desktop/*.mov -c:v libx264 -crf 23 -preset veryfast -an demo-video.mp4
```

## C. Live demo (in class) — same script, no recording

1. `docker compose up -d` (or already running)  
2. Show health green  
3. Ask the two questions above  
4. Show sources  
5. Show GitHub  

## D. Presentation

- File: `demo-assets/AI-ITI_Study_Assistant_Presentation.pptx`  
- Export/Print as PDF for the Drive folder  
- 9 slides, ~3–5 minutes if presented  

## E. Drive folder (must match the link you submitted)

```
Your Drive folder (Anyone with the link → Viewer)
├── ALI ezz - Intro to Deep Learning.png
├── ALI ezz - Computer Vision.png
├── demo-video.mp4
└── AI-ITI_Study_Assistant_Presentation.pdf
```

If the form is already submitted: open the response → **Edit** → re-submit so reviewers refresh the link contents (same folder URL works if you only add files).
