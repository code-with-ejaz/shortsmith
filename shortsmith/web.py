from __future__ import annotations

import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .pipeline import load_dotenv, process_video, validate_youtube_url

load_dotenv()
ROOT = Path(os.getenv("SHORTSMITH_OUTPUT", "output")).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="ShortSmith", docs_url=None, redoc_url=None)
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shortsmith")
jobs: dict[str, dict[str, Any]] = {}
lock = threading.Lock()


class JobRequest(BaseModel):
    url: str
    clips: int = Field(default=3, ge=1, le=5)
    min_seconds: int = Field(default=20, ge=10, le=60)
    max_seconds: int = Field(default=59, ge=10, le=60)
    whisper_model: str = Field(default_factory=lambda: os.getenv("SHORTSMITH_WHISPER_MODEL", "base"))
    # Optional[...] (not `str | None`): Pydantic evaluates model annotations at
    # class creation, and `str | None` fails on Python 3.9, the minimum supported.
    language: Optional[str] = None
    use_llm: bool = True


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ShortSmith — Make your shorts</title>
<style>
:root{color-scheme:dark;--bg:#101116;--panel:#191a22;--line:#2d2e39;--text:#f4f3f8;--muted:#a3a2b1;--accent:#bb8cff;--green:#85e0ba}*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 50% -20%,#342345,var(--bg) 60%);color:var(--text);font:16px/1.5 system-ui,sans-serif;min-height:100vh}.wrap{max-width:760px;margin:0 auto;padding:56px 20px 72px}header{text-align:center;margin-bottom:32px}.logo{font-size:13px;font-weight:800;letter-spacing:.18em;color:var(--accent);text-transform:uppercase}h1{font-size:clamp(34px,8vw,54px);line-height:1.04;margin:16px 0 12px;letter-spacing:-.04em}header p{color:var(--muted);max-width:530px;margin:auto}.card{background:color-mix(in srgb,var(--panel) 94%,transparent);border:1px solid var(--line);border-radius:18px;padding:24px;box-shadow:0 20px 70px #0005}label{display:block;font-size:14px;font-weight:700;margin-bottom:8px}input,select{width:100%;border:1px solid var(--line);background:#101116;color:var(--text);border-radius:10px;padding:13px;font:inherit}input:focus,select:focus{outline:2px solid var(--accent);border-color:transparent}.row{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:18px}.hint{font-size:12px;color:var(--muted);margin:7px 0 0}.check{display:flex;align-items:center;gap:9px;color:var(--muted);font-size:14px;margin:18px 0}.check input{width:auto;accent-color:var(--accent)}button{border:0;border-radius:10px;padding:13px 20px;background:var(--accent);color:#17101e;font:700 15px system-ui;cursor:pointer;width:100%}button:disabled{opacity:.55;cursor:wait}.privacy{color:var(--muted);font-size:12px;margin:14px 0 0;text-align:center}.status{margin-top:22px;display:none}.status.on{display:block}.statusline{display:flex;align-items:center;gap:10px;color:var(--muted);margin-bottom:10px}.dot{width:9px;height:9px;border-radius:50%;background:var(--accent);box-shadow:0 0 14px var(--accent)}.bar{height:5px;background:#30303a;border-radius:8px;overflow:hidden}.bar i{display:block;height:100%;width:35%;background:var(--accent);animation:move 1.6s ease-in-out infinite alternate}@keyframes move{to{transform:translateX(185%);width:45%}}.error{color:#ff9797}.clips{display:grid;gap:12px;margin-top:18px}.clip{border:1px solid var(--line);border-radius:12px;padding:16px}.clip h3{margin:0 0 6px;font-size:17px}.clip p{color:var(--muted);font-size:13px;margin:0 0 13px}.download{display:inline-block;width:auto;text-decoration:none;background:#24232e;color:var(--green);border:1px solid #3e3a4a;border-radius:8px;padding:9px 14px;font-weight:700;font-size:14px}.download:hover{background:#302d3b}.foot{text-align:center;color:var(--muted);font-size:12px;margin:24px 0} @media(max-width:520px){.wrap{padding-top:35px}.card{padding:18px}.row{grid-template-columns:1fr;gap:12px}}
</style></head><body><main class="wrap"><header><div class="logo">ShortSmith</div><h1>Your next short,<br>found for you.</h1><p>Paste a YouTube link. ShortSmith analyzes the transcript, picks compelling moments, and makes captioned vertical clips.</p></header>
<section class="card"><form id="form"><label for="url">YouTube video link</label><input id="url" name="url" type="url" required placeholder="https://www.youtube.com/watch?v=…"><p class="hint">One public video, not a playlist. We download audio first; video is fetched only for selected moments.</p><div class="row"><div><label for="clips">Number of clips</label><select id="clips"><option>1</option><option>2</option><option selected>3</option><option>4</option><option>5</option></select></div><div><label for="model">Transcription model</label><select id="model"><option>tiny</option><option selected>base</option><option>small</option><option>medium</option><option>large-v3</option></select></div></div><label class="check"><input id="llm" type="checkbox" checked> Use Claude transcript analysis if API key is configured</label><button id="submit" type="submit">Find moments &amp; create clips</button><p class="privacy">No source-video preview is shown. Only your finished clips are available to download.</p></form>
<div class="status" id="status"><div class="statusline"><span class="dot"></span><span id="message">Starting…</span></div><div class="bar" id="bar"><i></i></div><div class="clips" id="clipsOut"></div></div></section><p class="foot">Local processing · Audio-only analysis download · Finished MP4s up to 480p source quality</p></main>
<script>
const form=document.querySelector('#form'),button=document.querySelector('#submit'),statusBox=document.querySelector('#status'),message=document.querySelector('#message'),clipsOut=document.querySelector('#clipsOut'),bar=document.querySelector('#bar');let timer;
form.addEventListener('submit',async e=>{e.preventDefault();button.disabled=true;button.textContent='Starting…';statusBox.classList.add('on');clipsOut.innerHTML='';bar.style.display='block';message.className='';message.textContent='Starting…';try{const r=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:document.querySelector('#url').value,clips:Number(document.querySelector('#clips').value),whisper_model:document.querySelector('#model').value,use_llm:document.querySelector('#llm').checked})});const data=await r.json();if(!r.ok)throw Error(data.detail||'Could not start the job');poll(data.id)}catch(err){failed(err.message)}});
async function poll(id){try{const r=await fetch('/api/jobs/'+id);const data=await r.json();if(!r.ok)throw Error(data.detail||'Could not read job status');message.textContent=data.message||data.status;if(data.status==='done'){bar.style.display='none';button.disabled=false;button.textContent='Create more clips';clipsOut.innerHTML=data.clips.map(c=>`<article class="clip"><h3>${esc(c.hook)}</h3><p>${esc(c.reason)}</p><a class="download" href="/api/jobs/${id}/clips/${encodeURIComponent(c.filename)}">Download MP4</a></article>`).join('');return}if(data.status==='error')throw Error(data.message);timer=setTimeout(()=>poll(id),1800)}catch(err){failed(err.message)}}
function failed(text){clearTimeout(timer);bar.style.display='none';message.className='error';message.textContent=text;button.disabled=false;button.textContent='Try again'}function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PAGE


def run_job(job_id: str, request: JobRequest) -> None:
    def update(message: str) -> None:
        with lock:
            jobs[job_id]["message"] = message

    try:
        results = process_video(
            url=request.url,
            output_root=ROOT,
            clips=request.clips,
            min_seconds=request.min_seconds,
            max_seconds=request.max_seconds,
            whisper_model=request.whisper_model,
            language=request.language,
            use_llm=request.use_llm,
            progress=update,
            job_id=job_id,
        )
        with lock:
            jobs[job_id].update(status="done", message="Your clips are ready.", clips=results)
    except Exception as exc:
        with lock:
            jobs[job_id].update(status="error", message=str(exc)[:700], clips=[])


@app.post("/api/jobs", status_code=202)
def create_job(request: JobRequest) -> dict[str, str]:
    try:
        request.url = validate_youtube_url(request.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if request.min_seconds > request.max_seconds:
        raise HTTPException(status_code=400, detail="Minimum duration must not exceed maximum duration.")
    if request.whisper_model not in {"tiny", "base", "small", "medium", "large-v3"}:
        raise HTTPException(status_code=400, detail="Choose a supported Whisper model.")
    job_id = uuid.uuid4().hex
    with lock:
        jobs[job_id] = {"status": "working", "message": "Queued…", "clips": []}
    executor.submit(run_job, job_id, request)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return dict(job)


@app.get("/api/jobs/{job_id}/clips/{filename}")
def get_clip(job_id: str, filename: str) -> FileResponse:
    if not re.fullmatch(r"[a-zA-Z0-9_-]+\.mp4", filename):
        raise HTTPException(status_code=400, detail="Invalid clip filename.")
    with lock:
        job = jobs.get(job_id)
        if job is None or job["status"] != "done":
            raise HTTPException(status_code=404, detail="Finished clip not found.")
        allowed = {clip["filename"] for clip in job["clips"]}
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="Finished clip not found.")
    path = (ROOT / job_id / filename).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise HTTPException(status_code=404, detail="Finished clip not found.")
    return FileResponse(path, media_type="video/mp4", filename=filename)


def main() -> None:
    import uvicorn
    uvicorn.run("shortsmith.web:app", host="127.0.0.1", port=int(os.getenv("PORT", "8000")), reload=False)


if __name__ == "__main__":
    main()
