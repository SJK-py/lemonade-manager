# lemonade_manager.py
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
import os
import httpx
import uvicorn
import json
from pathlib import Path
from typing import Dict, Any, Optional, Set

LEMONADE_BASE = os.getenv("LEMONADE_BASE", "http://localhost:8000")
DEFAULTS_FILE = Path(os.getenv("DEFAULTS_FILE", "model_defaults.json"))
LEMONADE_MANAGER_HOST = os.getenv("LEMONADE_MANAGER_HOST", "0.0.0.0")
LEMONADE_MANAGER_PORT = int(os.getenv("LEMONADE_MANAGER_PORT", "9000"))

app = FastAPI(title="Lemonade Model Manager")


# ---------- defaults & disabled storage ----------

def load_defaults() -> Dict[str, Dict[str, Any]]:
    if DEFAULTS_FILE.exists():
        try:
            return json.loads(DEFAULTS_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_defaults(data: Dict[str, Dict[str, Any]]) -> None:
    DEFAULTS_FILE.write_text(json.dumps(data, indent=2))


def get_model_defaults(model_name: str) -> Dict[str, Any]:
    all_defs = load_defaults()
    return all_defs.get("models", {}).get(model_name, {})


def set_model_defaults(
    model_name: str, 
    ctx_size: Optional[int], 
    llamacpp_args: Optional[str],
    llamacpp_backend: Optional[str]
) -> None:
    all_defs = load_defaults()
    models_def = all_defs.setdefault("models", {})
    entry: Dict[str, Any] = {}
    if ctx_size is not None:
        entry["ctx_size"] = ctx_size
    if llamacpp_args is not None and llamacpp_args.strip():
        entry["llamacpp_args"] = llamacpp_args.strip()
    if llamacpp_backend is not None and llamacpp_backend.strip():
        entry["llamacpp_backend"] = llamacpp_backend.strip()
        
    models_def[model_name] = entry
    save_defaults(all_defs)


def get_disabled_models() -> Set[str]:
    all_defs = load_defaults()
    disabled = all_defs.get("disabled", [])
    if isinstance(disabled, list):
        return set(disabled)
    return set()


def set_disabled(model_name: str, disabled: bool) -> None:
    all_defs = load_defaults()
    disabled_list = set(all_defs.get("disabled", []))
    if disabled:
        disabled_list.add(model_name)
    else:
        disabled_list.discard(model_name)
    all_defs["disabled"] = sorted(disabled_list)
    save_defaults(all_defs)


# ---------- lemonade helpers ----------

async def get_models():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{LEMONADE_BASE}/api/v1/models")
        r.raise_for_status()
        return r.json()


async def get_health():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{LEMONADE_BASE}/api/v1/health")
        r.raise_for_status()
        return r.json()


async def get_stats():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{LEMONADE_BASE}/api/v1/stats")
        if r.status_code != 200:
            return None
        return r.json()


async def do_load(
    model_name: str, 
    ctx_size: Optional[int], 
    llamacpp_args: Optional[str], 
    llamacpp_backend: Optional[str]
):
    payload: Dict[str, Any] = {"model_name": model_name}
    if ctx_size:
        payload["ctx_size"] = ctx_size
    if llamacpp_args and llamacpp_args.strip():
        payload["llamacpp_args"] = llamacpp_args.strip()
    if llamacpp_backend and llamacpp_backend.strip():
        payload["llamacpp_backend"] = llamacpp_backend.strip()

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{LEMONADE_BASE}/api/v1/load", json=payload)
        r.raise_for_status()


async def do_unload_model(model_name: str):
    payload = {"model_name": model_name}
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{LEMONADE_BASE}/api/v1/unload", json=payload)
        r.raise_for_status()


# ---------- UI ----------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    models = await get_models()
    health = await get_health()
    stats = await get_stats()

    data = models.get("data", [])

    loaded_ids = set()
    if isinstance(health, dict):
        for entry in health.get("all_models_loaded", []):
            mid = entry.get("model_name")
            if mid:
                loaded_ids.add(mid)

    disabled_models = get_disabled_models()

    def esc(s: Any):
        return (str(s) or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    rows = []
    for m in data:
        mid = m.get("id")
        recipe = m.get("recipe", "") or ""
        downloaded = m.get("downloaded", True) if "downloaded" in m else True
        is_loaded = mid in loaded_ids
        is_disabled = mid in disabled_models
        
        # Check if recipe implies llamacpp
        is_llamacpp = "llamacpp" in recipe.lower() or "gguf" in mid.lower() or "gguf" in recipe.lower()

        defaults = get_model_defaults(mid)
        def_ctx = defaults.get("ctx_size", "")
        def_args = defaults.get("llamacpp_args", "")
        def_backend = defaults.get("llamacpp_backend", "")

        row_style = "opacity:0.45;" if is_disabled else ""
        form_id = f"form-{mid}"

        # 1. ID Column
        if is_disabled:
            id_html = f"""
            <code>{esc(mid)}</code>
            <form method="post" action="/disable" style="display:inline; margin-left:0.4rem;">
              <input type="hidden" name="model_name" value="{esc(mid)}">
              <input type="hidden" name="disabled" value="0">
              <button type="submit">Enable</button>
            </form>
            """
        else:
            id_html = f"""
            <code>{esc(mid)}</code>
            <form method="post" action="/disable" style="display:inline; margin-left:0.4rem;">
              <input type="hidden" name="model_name" value="{esc(mid)}">
              <input type="hidden" name="disabled" value="1">
              <button type="submit">Disable</button>
            </form>
            """

        # 2. Recipe Column (with backend input if llamacpp)
        if is_llamacpp and not is_disabled:
            # Note: form={form_id} connects this input to the main form in Actions column
            recipe_html = f"""
            <div>{esc(recipe)}</div>
            <div style="margin-top:0.2rem;">
               <input form="{form_id}" type="text" name="llamacpp_backend" placeholder="backend (e.g. vulkan)" style="width:9em; font-size:0.75rem;" value="{esc(def_backend)}">
            </div>
            """
        else:
            recipe_html = esc(recipe)

        # 3. Loaded Column
        if is_loaded:
            loaded_html = f"""
            <div>â</div>
            <form method="post" action="/unload/model" style="margin-top:0.2rem;">
              <input type="hidden" name="model_name" value="{esc(mid)}">
              <button type="submit">Unload</button>
            </form>
            """
        else:
            loaded_html = "â"

        # 4. Actions / Defaults Column
        if is_disabled:
            actions_html = f"""
            <div style="font-size:0.8rem; color:#9ca3af;">
              Model disabled.
            </div>
            <div style="font-size:0.75rem; color:#9ca3af; margin-top:0.1rem;">
              Default ctx: <code>{esc(def_ctx)}</code>,
              args: <code>{esc(def_args)}</code>
              {f', backend: <code>{esc(def_backend)}</code>' if is_llamacpp else ''}
            </div>
            """
        else:
            # Use one main form for the row. 
            # The backend input in the Recipe column links here via form="{form_id}".
            actions_html = f"""
            <form id="{form_id}" method="post" style="display:inline;">
                <input type="hidden" name="model_name" value="{esc(mid)}">
                
                <div style="margin-bottom:0.25rem;">
                  <button type="submit" formaction="/defaults/load">Load default</button>
                  <span style="font-size:0.75rem; color:#9ca3af; margin-left:0.4rem;">
                    ctx: <code>{esc(def_ctx)}</code>,
                    args: <code>{esc(def_args)}</code>
                  </span>
                </div>
                
                <div>
                  <input type="number" name="ctx_size" placeholder="ctx" min="1024" step="1024" style="width:5em;">
                  <input type="text" name="llamacpp_args" placeholder="llamacpp_args" style="width:12em;">
                  
                  <button type="submit" formaction="/load">Load custom</button>
                  <button type="submit" formaction="/defaults/set" style="margin-left:0.2rem;">Set default</button>
                </div>
            </form>
            """

        rows.append(f"""
        <tr style="{row_style}">
          <td>{id_html}</td>
          <td>{recipe_html}</td>
          <td>{'â' if downloaded else 'â'}</td>
          <td>{loaded_html}</td>
          <td>{actions_html}</td>
        </tr>
        """)

    stats_html = ""
    if stats:
        stats_html = f"""
        <h2>Last request stats</h2>
        <pre>{esc(str(stats))}</pre>
        """

    body = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Lemonade Model Manager</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background:#0b0b0e; color:#e5e7eb; padding:1.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-top:1rem; }}
    th, td {{ border: 1px solid #374151; padding: 0.35rem 0.5rem; font-size: 0.9rem; vertical-align: top; }}
    th {{ background:#111827; }}
    tr:nth-child(even) {{ background:#111827; }}
    button {{ background:#4b5563; color:white; border:none; padding:0.25rem 0.55rem; border-radius:4px; cursor:pointer; font-size:0.8rem; }}
    button:hover {{ background:#6b7280; }}
    input {{ background:#111827; border:1px solid #374151; color:#e5e7eb; border-radius:4px; padding:0.1rem 0.25rem; font-size:0.8rem; }}
    h1, h2 {{ margin-top:0.5rem; }}
    code {{ background:#111827; padding:0.1rem 0.25rem; border-radius:3px; }}
    .toolbar {{ margin-bottom:1rem; display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap; }}
  </style>
</head>
<body>
  <h1>Lemonade Model Manager</h1>
  <div class="toolbar">
    <form method="post" action="/unload">
      <button type="submit">Unload all models</button>
    </form>
    <span>Current model: <code>{esc(health.get('model_loaded', '') if isinstance(health, dict) else '')}</code></span>
  </div>

  <table>
    <thead>
      <tr>
        <th>Model ID</th>
        <th>Recipe</th>
        <th>Downloaded</th>
        <th>Loaded</th>
        <th>Actions / Defaults</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>

  {stats_html}
</body>
</html>
"""
    return HTMLResponse(body)


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


# ---------- actions ----------

@app.post("/load")
async def load_model(
    model_name: str = Form(...),
    ctx_size: Optional[int] = Form(None),
    llamacpp_args: Optional[str] = Form(None),
    llamacpp_backend: Optional[str] = Form(None),
):
    # This action handles "Load custom".
    # Use whatever the user typed in the inputs.
    await do_load(model_name, ctx_size, llamacpp_args, llamacpp_backend)
    return RedirectResponse(url="/", status_code=303)


@app.post("/defaults/load")
async def load_model_defaults(
    model_name: str = Form(...),
    llamacpp_backend: Optional[str] = Form(None)
):
    # This action handles "Load default".
    # Ignore the explicit ctx/args text boxes (user didn't click Load Custom).
    # Load stored defaults for ctx/args.
    # BUT, DO respect the backend input if the user typed one, overriding the stored default backend.
    
    defaults = get_model_defaults(model_name)
    ctx_size = defaults.get("ctx_size")
    llamacpp_args = defaults.get("llamacpp_args")
    
    # If user typed a backend in the box, use it. Otherwise use stored default backend.
    final_backend = llamacpp_backend if (llamacpp_backend and llamacpp_backend.strip()) else defaults.get("llamacpp_backend")
    
    await do_load(model_name, ctx_size, llamacpp_args, final_backend)
    return RedirectResponse(url="/", status_code=303)


@app.post("/defaults/set")
async def set_defaults(
    model_name: str = Form(...),
    ctx_size: Optional[int] = Form(None),
    llamacpp_args: Optional[str] = Form(None),
    llamacpp_backend: Optional[str] = Form(None),
):
    # This action handles "Set default".
    # It saves whatever values are currently in the inputs (including backend) as the new defaults.
    set_model_defaults(model_name, ctx_size, llamacpp_args, llamacpp_backend)
    return RedirectResponse(url="/", status_code=303)


@app.post("/unload")
async def unload_all_models():
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{LEMONADE_BASE}/api/v1/unload")
        r.raise_for_status()
    return RedirectResponse(url="/", status_code=303)


@app.post("/unload/model")
async def unload_one_model(model_name: str = Form(...)):
    await do_unload_model(model_name)
    return RedirectResponse(url="/", status_code=303)


@app.post("/disable")
async def disable_model(
    model_name: str = Form(...),
    disabled: str = Form(...),  # "1" or "0"
):
    set_disabled(model_name, disabled == "1")
    return RedirectResponse(url="/", status_code=303)


if __name__ == "__main__":
    uvicorn.run("lemonade_manager:app", host=LEMONADE_MANAGER_HOST, port=LEMONADE_MANAGER_PORT, reload=False)
