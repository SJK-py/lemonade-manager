# lemonade_manager.py
# v2.0
# =============================================================================
# v1.0 -> v2.0
# - Added env var LEMONADE_KEY, MANAGER_HOST, TIMEOUT_PULL
# - Implemented POST /api/v1/pull, POST /api/v1/delete
# - Refactored httpx calls
# - Reinforced comments
# =============================================================================
import os
import json
import uvicorn
import httpx
import asyncio
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from pathlib import Path
from typing import Dict, Any, Optional, Set, AsyncGenerator

# =============================================================================
# CONFIGURATION
# =============================================================================

# The base URL of the Lemonade Server instance
LEMONADE_BASE = os.getenv("LEMONADE_BASE", "http://localhost:8000")

# Optional API Key for Lemonade Server (Bearer Token)
# If set, this will be added to the Authorization header of every request.
LEMONADE_KEY = os.getenv("LEMONADE_KEY", "")

# Host interface to bind the Manager UI to.
# Default: 0.0.0.0
MANAGER_HOST = os.getenv("MANAGER_HOST", "0.0.0.0")

# Port for this Manager UI
MANAGER_PORT = int(os.getenv("MANAGER_PORT", "9000"))

# Timeout settings (in seconds)
TIMEOUT_LOAD = float(os.getenv("TIMEOUT_LOAD", "120.0"))   # Loading a model
TIMEOUT_LIGHT = float(os.getenv("TIMEOUT_LIGHT", "10.0"))  # Stats/Health/Delete
TIMEOUT_PULL = float(os.getenv("TIMEOUT_PULL", "3600.0"))  # Pulling/Downloading (can take a long time)

# Path to the native Lemonade Server configuration file.
# Default: ~/.cache/lemonade/recipe_options.json
_default_recipe_path = Path("~/.cache/lemonade/recipe_options.json").expanduser()
RECIPE_FILE = Path(os.getenv("RECIPE_FILE", str(_default_recipe_path)))

# Path to local preferences for this manager (stores 'disabled' list)
PREFS_FILE = Path(os.getenv("PREFS_FILE", "manager_prefs.json"))

# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(title="Lemonade Model Manager")

# =============================================================================
# STORAGE: NATIVE SERVER CONFIG (recipe_options.json)
# =============================================================================

def load_recipe_options() -> Dict[str, Dict[str, Any]]:
    """
    Loads the native lemonade-server configuration file.
    This allows the manager to sync with settings used by the CLI/Server directly.
    """
    if RECIPE_FILE.exists():
        try:
            return json.loads(RECIPE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_recipe_options(data: Dict[str, Dict[str, Any]]) -> None:
    """
    Writes to the native lemonade-server configuration file.
    Ensures the directory exists before writing.
    """
    if not RECIPE_FILE.parent.exists():
        try:
            RECIPE_FILE.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass 

    RECIPE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

def get_model_options(model_name: str) -> Dict[str, Any]:
    """Retrieves options for a specific model from the global recipe file."""
    all_opts = load_recipe_options()
    return all_opts.get(model_name, {})

def set_model_options(
    model_name: str, 
    ctx_size: Optional[int], 
    llamacpp_args: Optional[str],
    llamacpp_backend: Optional[str]
) -> None:
    """
    Updates options for a specific model.
    Removes empty keys to keep the JSON clean.
    """
    all_opts = load_recipe_options()
    entry = all_opts.get(model_name, {})

    if ctx_size is not None:
        entry["ctx_size"] = ctx_size

    if llamacpp_args is not None:
        clean_args = llamacpp_args.strip()
        if clean_args:
            entry["llamacpp_args"] = clean_args
        elif "llamacpp_args" in entry:
            del entry["llamacpp_args"]

    if llamacpp_backend is not None:
        clean_backend = llamacpp_backend.strip()
        if clean_backend:
            entry["llamacpp_backend"] = clean_backend
        elif "llamacpp_backend" in entry:
            del entry["llamacpp_backend"]

    if entry:
        all_opts[model_name] = entry
    elif model_name in all_opts:
        del all_opts[model_name]

    save_recipe_options(all_opts)

# =============================================================================
# STORAGE: LOCAL MANAGER PREFS (manager_prefs.json)
# =============================================================================

def get_disabled_models() -> Set[str]:
    """Reads the local list of 'disabled' (hidden) models."""
    if PREFS_FILE.exists():
        try:
            data = json.loads(PREFS_FILE.read_text(encoding="utf-8"))
            return set(data.get("disabled", []))
        except Exception:
            return set()
    return set()

def set_disabled(model_name: str, disabled: bool) -> None:
    """Updates the local list of disabled models."""
    current_disabled = get_disabled_models()
    if disabled:
        current_disabled.add(model_name)
    else:
        current_disabled.discard(model_name)

    data = {"disabled": sorted(list(current_disabled))}
    PREFS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

# =============================================================================
# API CLIENT HELPERS
# =============================================================================

def get_headers() -> Dict[str, str]:
    """Generates headers for Lemonade Server requests, including Auth."""
    headers = {}
    if LEMONADE_KEY:
        headers["Authorization"] = f"Bearer {LEMONADE_KEY}"
    return headers

async def get_models():
    """Fetches list of available models."""
    async with httpx.AsyncClient(timeout=TIMEOUT_LIGHT) as client:
        r = await client.get(f"{LEMONADE_BASE}/api/v1/models", headers=get_headers())
        r.raise_for_status()
        return r.json()

async def get_health():
    """Fetches server health status (currently loaded models)."""
    async with httpx.AsyncClient(timeout=TIMEOUT_LIGHT) as client:
        r = await client.get(f"{LEMONADE_BASE}/api/v1/health", headers=get_headers())
        r.raise_for_status()
        return r.json()

async def get_stats():
    """Fetches performance stats."""
    async with httpx.AsyncClient(timeout=TIMEOUT_LIGHT) as client:
        try:
            r = await client.get(f"{LEMONADE_BASE}/api/v1/stats", headers=get_headers())
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    return None

async def do_load(
    model_name: str,
    ctx_size: Optional[int],
    llamacpp_args: Optional[str],
    llamacpp_backend: Optional[str]
):
    """Sends the load command to Lemonade Server."""
    payload: Dict[str, Any] = {"model_name": model_name}

    if ctx_size:
        payload["ctx_size"] = ctx_size
    if llamacpp_args and llamacpp_args.strip():
        payload["llamacpp_args"] = llamacpp_args.strip()
    if llamacpp_backend and llamacpp_backend.strip():
        payload["llamacpp_backend"] = llamacpp_backend.strip()

    async with httpx.AsyncClient(timeout=TIMEOUT_LOAD) as client:
        r = await client.post(f"{LEMONADE_BASE}/api/v1/load", json=payload, headers=get_headers())
        r.raise_for_status()

async def do_unload_model(model_name: str):
    """Unloads a specific model."""
    payload = {"model_name": model_name}
    async with httpx.AsyncClient(timeout=TIMEOUT_LIGHT) as client:
        r = await client.post(f"{LEMONADE_BASE}/api/v1/unload", json=payload, headers=get_headers())
        r.raise_for_status()

async def do_delete_model(model_name: str):
    """
    Deletes a model from local storage.
    Note: Lemonade Server will unload it first if it is running.
    """
    payload = {"model_name": model_name}
    async with httpx.AsyncClient(timeout=TIMEOUT_LIGHT) as client:
        r = await client.post(f"{LEMONADE_BASE}/api/v1/delete", json=payload, headers=get_headers())
        r.raise_for_status()

# =============================================================================
# UI GENERATION
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Gather data from Lemonade Server
    try:
        models = await get_models()
        health = await get_health()
        stats = await get_stats()
    except Exception as e:
        # Graceful failure if server is down or Auth failed
        return HTMLResponse(f"""
            <html><body style="font-family:sans-serif; background:#0b0b0e; color:#e5e7eb; padding:2rem;">
            <h1>Connection Error</h1>
            <p>Could not connect to Lemonade Server at <code>{LEMONADE_BASE}</code></p>
            <p><strong>Note:</strong> If the server requires an API Key, ensure <code>LEMONADE_KEY</code> is set.</p>
            <pre>{str(e)}</pre>
            <p><a href="/" style="color:#60a5fa;">Retry</a></p>
            </body></html>
        """)

    data = models.get("data", [])

    # Identify currently loaded models
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
        downloaded = m.get("downloaded", True)
        is_loaded = mid in loaded_ids
        is_disabled = mid in disabled_models

        is_llamacpp = "llamacpp" in recipe.lower() or "gguf" in mid.lower() or "gguf" in recipe.lower()

        defaults = get_model_options(mid)
        def_ctx = defaults.get("ctx_size", "")
        def_args = defaults.get("llamacpp_args", "")
        def_backend = defaults.get("llamacpp_backend", "")

        row_class = "disabled-row" if is_disabled else ""
        form_id = f"form-{mid}"

        # ------------------ 1. ID / Enable / Delete ------------------
        # Added Delete button here (v2.0)
        delete_btn_html = f"""
        <button type="button" class="btn-xs btn-outline btn-trash" 
            onclick="showDeleteConfirm('{esc(mid)}')" title="Delete Model">🗑️</button>
        """

        if is_disabled:
            id_html = f"""
            <div class="model-id">{esc(mid)}</div>
            <div class="btn-group">
                <form method="post" action="/disable" class="inline-form">
                  <input type="hidden" name="model_name" value="{esc(mid)}">
                  <input type="hidden" name="disabled" value="0">
                  <button type="submit" class="btn-xs">Enable</button>
                </form>
                {delete_btn_html}
            </div>
            """
        else:
            id_html = f"""
            <div class="model-id">{esc(mid)}</div>
            <div class="btn-group">
                <form method="post" action="/disable" class="inline-form">
                  <input type="hidden" name="model_name" value="{esc(mid)}">
                  <input type="hidden" name="disabled" value="1">
                  <button type="submit" class="btn-xs btn-outline">Disable</button>
                </form>
                {delete_btn_html}
            </div>
            """

        # ------------------ 2. Recipe / Backend ------------------
        if is_llamacpp and not is_disabled:
            recipe_html = f"""
            <div class="recipe-text">{esc(recipe)}</div>
            <div class="backend-wrapper">
               <input form="{form_id}" type="text" name="llamacpp_backend" 
                      placeholder="backend (e.g. vulkan)" 
                      class="input-backend"
                      value="{esc(def_backend)}">
            </div>
            """
        else:
            recipe_html = f'<div class="recipe-text">{esc(recipe)}</div>'

        # ------------------ 3. Status ------------------
        if is_loaded:
            status_html = f"""
            <div class="status-badge loaded">Running</div>
            <form method="post" action="/unload/model" class="mt-1" onsubmit="showLoading()">
              <input type="hidden" name="model_name" value="{esc(mid)}">
              <button type="submit" class="btn-xs btn-red">Unload</button>
            </form>
            """
        else:
            status_html = '<div class="status-badge">Stopped</div>'

        # ------------------ 4. Actions ------------------
        if is_disabled:
            actions_html = f"""
            <div class="text-muted text-sm">
              <em>Model is hidden. Enable to configure.</em>
            </div>
            """
        else:
            actions_html = f"""
            <form id="{form_id}" method="post" class="action-form" onsubmit="showLoading()">
                <input type="hidden" name="model_name" value="{esc(mid)}">

                <div class="action-row">
                  <button type="submit" formaction="/defaults/load" class="btn-primary" title="Load using parameters from recipe_options.json">
                    Load (Default)
                  </button>
                  <div class="info-text">
                    Saved: <strong>{esc(def_ctx) if def_ctx else 'default'}</strong> ctx, 
                    <strong>{esc(def_args) if def_args else 'none'}</strong> args
                  </div>
                </div>

                <div class="action-row mt-1">
                  <input type="number" name="ctx_size" placeholder="ctx size" min="1024" step="1024" class="input-ctx" value="{esc(def_ctx)}">
                  <input type="text" name="llamacpp_args" placeholder="args (e.g. -np 4)" class="input-args" value="{esc(def_args)}">

                  <div class="btn-group">
                    <button type="submit" formaction="/load" class="btn-secondary">Load Custom</button>
                    <button type="submit" formaction="/defaults/set" class="btn-save" title="Save these settings to recipe_options.json">Save</button>
                  </div>
                </div>
            </form>
            """

        rows.append(f"""
        <tr class="{row_class}">
          <td>{id_html}</td>
          <td>{recipe_html}</td>
          <td class="center-text">{'✅' if downloaded else '❌'}</td>
          <td>{status_html}</td>
          <td>{actions_html}</td>
        </tr>
        """)

    # Stats Section
    stats_html = ""
    if stats:
        stats_html = f"""
        <div class="stats-container">
            <h2>Last Request Stats</h2>
            <pre>{esc(str(stats))}</pre>
        </div>
        """

    loaded_model_name = health.get('model_loaded', 'None') if isinstance(health, dict) else 'Unknown'

    # ------------------ HTML BODY ------------------
    body = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Lemonade Manager</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
        --bg-body: #0b0b0e;
        --bg-panel: #111827;
        --border: #374151;
        --text-main: #e5e7eb;
        --text-muted: #9ca3af;
        --primary: #2563eb;
        --primary-hover: #1d4ed8;
        --danger: #7f1d1d;
        --success: #065f46;
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: system-ui, -apple-system, sans-serif; background: var(--bg-body); color: var(--text-main); margin: 0; padding: 1.5rem; }}

    /* Layout */
    .toolbar {{ display: flex; justify-content: space-between; align-items: center; background: var(--bg-panel); padding: 1rem; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 1.5rem; flex-wrap: wrap; gap: 1rem; }}
    .section-container {{ margin-top: 2rem; background: var(--bg-panel); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--border); }}
    .stats-container {{ margin-top: 2rem; background: var(--bg-panel); padding: 1rem; border-radius: 8px; border: 1px solid var(--border); }}

    /* Tables */
    table {{ width: 100%; border-collapse: separate; border-spacing: 0; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 0.75rem 1rem; border-bottom: 1px solid var(--border); vertical-align: top; }}
    th {{ background: #1f2937; text-align: left; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:nth-child(even) {{ background: #131b2e; }}
    .disabled-row {{ opacity: 0.5; background: #0f1115 !important; }}
    .center-text {{ text-align: center; }}

    /* Typography */
    h1 {{ margin: 0; font-size: 1.5rem; display: flex; align-items: center; gap: 0.5rem; }}
    h2 {{ margin-top: 0; font-size: 1.25rem; color: var(--primary); }}
    code {{ font-family: 'Menlo', 'Monaco', monospace; background: #1f2937; padding: 0.2rem 0.4rem; border-radius: 4px; color: #60a5fa; font-size: 0.9em; }}
    .text-muted {{ color: var(--text-muted); }}
    .text-sm {{ font-size: 0.8rem; }}
    .model-id {{ font-family: monospace; font-weight: bold; margin-bottom: 0.5rem; word-break: break-all; }}

    /* Forms & Inputs */
    input {{ background: #1f2937; border: 1px solid #4b5563; color: var(--text-main); border-radius: 4px; padding: 0.35rem 0.5rem; font-size: 0.85rem; }}
    input:focus {{ outline: 2px solid var(--primary); border-color: transparent; }}
    input:disabled {{ opacity: 0.5; cursor: not-allowed; }}
    .input-ctx {{ width: 6rem; }}
    .input-args {{ width: 12rem; }}
    .input-backend {{ width: 100%; margin-top: 0.25rem; }}
    .form-group {{ margin-bottom: 0.75rem; }}
    .form-label {{ display: block; font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.25rem; }}
    .form-control {{ width: 100%; }}
    .checkbox-wrapper {{ display: flex; align-items: center; gap: 0.5rem; }}

    /* Buttons */
    button {{ cursor: pointer; border: none; border-radius: 4px; padding: 0.35rem 0.75rem; font-size: 0.85rem; font-weight: 500; transition: all 0.15s; color: white; }}
    .btn-primary {{ background: var(--primary); }}
    .btn-primary:hover {{ background: var(--primary-hover); }}
    .btn-secondary {{ background: #4b5563; }}
    .btn-secondary:hover {{ background: #6b7280; }}
    .btn-save {{ background: #059669; }}
    .btn-save:hover {{ background: #047857; }}
    .btn-red {{ background: #991b1b; }}
    .btn-red:hover {{ background: #b91c1c; }}
    .btn-outline {{ background: transparent; border: 1px solid #4b5563; }}
    .btn-outline:hover {{ background: #374151; }}
    .btn-xs {{ padding: 0.2rem 0.5rem; font-size: 0.75rem; }}
    .btn-trash {{ color: #ef4444; border-color: #7f1d1d; }}
    .btn-trash:hover {{ background: #7f1d1d; color: white; }}

    /* Flex Utilities */
    .mt-1 {{ margin-top: 0.5rem; }}
    .action-row {{ display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; }}
    .btn-group {{ display: flex; gap: 0.25rem; }}
    .info-text {{ font-size: 0.75rem; color: var(--text-muted); }}
    .status-badge {{ display: inline-block; padding: 0.2rem 0.6rem; border-radius: 99px; font-size: 0.75rem; font-weight: bold; background: #374151; color: #9ca3af; }}
    .status-badge.loaded {{ background: #064e3b; color: #6ee7b7; border: 1px solid #059669; }}

    /* Modal / Dialog */
    .modal {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 9000; align-items: center; justify-content: center; }}
    .modal-content {{ background: var(--bg-panel); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--border); width: 90%; max-width: 500px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
    .modal-actions {{ display: flex; justify-content: flex-end; gap: 0.5rem; margin-top: 1.5rem; }}
    .modal-title {{ font-size: 1.25rem; font-weight: bold; margin-bottom: 1rem; color: #f3f4f6; }}
    
    /* Loading Overlay */
    #loading-overlay {{
        display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0,0,0,0.8); z-index: 9999;
        align-items: center; justify-content: center; flex-direction: column;
    }}
    .spinner {{
        width: 40px; height: 40px; border: 4px solid #374151; border-top: 4px solid #3b82f6; border-radius: 50%;
        animation: spin 1s linear infinite; margin-bottom: 1rem;
    }}
    @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}

    /* Progress Log */
    #pull-progress {{
        margin-top: 1rem; background: #000; padding: 1rem; border-radius: 4px; 
        font-family: monospace; font-size: 0.8rem; height: 150px; overflow-y: auto;
        border: 1px solid var(--border); display: none;
    }}
    .progress-line {{ margin-bottom: 0.25rem; border-bottom: 1px solid #333; padding-bottom: 0.25rem; }}

  </style>
  <script>
    function showLoading() {{
        document.getElementById('loading-overlay').style.display = 'flex';
    }}

    // --- Delete Feature Scripts ---
    let deleteModelTarget = "";
    
    function showDeleteConfirm(modelName) {{
        deleteModelTarget = modelName;
        document.getElementById('delete-target-name').innerText = modelName;
        document.getElementById('delete-modal').style.display = 'flex';
    }}

    function closeDeleteModal() {{
        document.getElementById('delete-modal').style.display = 'none';
        deleteModelTarget = "";
    }}

    function confirmDelete() {{
        if(!deleteModelTarget) return;
        const form = document.createElement('form');
        form.method = 'POST';
        form.action = '/delete_model';
        const input = document.createElement('input');
        input.type = 'hidden';
        input.name = 'model_name';
        input.value = deleteModelTarget;
        form.appendChild(input);
        document.body.appendChild(form);
        showLoading();
        form.submit();
    }}

    // --- Pull Feature Scripts ---
    function toggleMmproj() {{
        const chk = document.getElementById('mmproj_enabled');
        const txt = document.getElementById('mmproj_input');
        txt.disabled = !chk.checked;
        if (!chk.checked) txt.value = "";
    }}

    function showPullConfirm(e) {{
        e.preventDefault();
        // Gather values for display
        const mName = document.getElementById('pull_model_name').value;
        const mCheck = document.getElementById('pull_checkpoint').value;
        const mRecipe = document.getElementById('pull_recipe').value;
        
        document.getElementById('conf_m_name').innerText = mName;
        document.getElementById('conf_m_check').innerText = mCheck;
        document.getElementById('conf_m_recipe').innerText = mRecipe;
        
        document.getElementById('pull-modal').style.display = 'flex';
    }}

    function closePullModal() {{
        document.getElementById('pull-modal').style.display = 'none';
    }}

    async function executePull() {{
        // Switch view to progress
        document.getElementById('pull-actions').style.display = 'none';
        document.getElementById('pull-status-msg').innerText = "Initializing download... DO NOT CLOSE THIS WINDOW.";
        const logBox = document.getElementById('pull-progress');
        logBox.style.display = 'block';
        logBox.innerHTML = "<div>Starting stream...</div>";

        // Prepare Form Data
        const formData = new FormData(document.getElementById('pull-form'));
        
        try {{
            const response = await fetch('/pull/stream', {{
                method: 'POST',
                body: formData
            }});

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            while (true) {{
                const {{ done, value }} = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                // SSE format usually sends "event: ... \n data: ... \n\n"
                // Simple parsing for display:
                const lines = chunk.split('\\n');
                let bufferHtml = "";
                
                lines.forEach(line => {{
                    if(line.startsWith('data: ')) {{
                        try {{
                            const jsonData = JSON.parse(line.substring(6));
                            if (jsonData.percent) {{
                                bufferHtml += `<div class="progress-line">Progress: ${{jsonData.percent}}% - File index: ${{jsonData.file_index || '?'}}</div>`;
                            }} else if (jsonData.error) {{
                                bufferHtml += `<div style="color:red">Error: ${{jsonData.error}}</div>`;
                            }} else {{
                                bufferHtml += `<div>${{JSON.stringify(jsonData)}}</div>`;
                            }}
                        }} catch (e) {{ 
                           // ignore incomplete json chunks
                        }}
                    }}
                }});
                
                if (bufferHtml) {{
                    logBox.innerHTML += bufferHtml;
                    logBox.scrollTop = logBox.scrollHeight;
                }}
            }}
            
            // Done
            logBox.innerHTML += "<div style='color:#6ee7b7; font-weight:bold; margin-top:10px;'>Process Complete. Reloading...</div>";
            setTimeout(() => window.location.reload(), 2000);

        }} catch (err) {{
            logBox.innerHTML += `<div style="color:red">Network Error: ${{err}}</div>`;
            document.getElementById('pull-actions').style.display = 'flex';
        }}
    }}
  </script>
</head>
<body>

  <div id="loading-overlay">
    <div class="spinner"></div>
    <div>Processing... please wait</div>
  </div>

  <div id="delete-modal" class="modal">
    <div class="modal-content">
        <div class="modal-title" style="color:#ef4444">⚠️ Delete Model?</div>
        <p>Are you sure you want to delete <strong><span id="delete-target-name"></span></strong>?</p>
        <p>This will remove the files from disk. <span style="color:#ef4444; font-weight:bold;">This action is irreversible.</span></p>
        <div class="modal-actions">
            <button onclick="closeDeleteModal()" class="btn-secondary">Cancel</button>
            <button onclick="confirmDelete()" class="btn-red">Yes, Delete</button>
        </div>
    </div>
  </div>

  <div id="pull-modal" class="modal">
    <div class="modal-content">
        <div class="modal-title">Confirm Download</div>
        <div id="pull-status-msg">
            <p>You are about to pull the following model:</p>
            <ul>
                <li><strong>Name:</strong> <span id="conf_m_name"></span></li>
                <li><strong>Checkpoint:</strong> <span id="conf_m_check"></span></li>
                <li><strong>Recipe:</strong> <span id="conf_m_recipe"></span></li>
            </ul>
            <p class="text-sm text-muted">Timeout set to: {int(TIMEOUT_PULL)}s. Large models may take a long time.</p>
        </div>
        
        <div id="pull-progress"></div>

        <div class="modal-actions" id="pull-actions">
            <button onclick="closePullModal()" class="btn-secondary">Cancel</button>
            <button onclick="executePull()" class="btn-primary">Yes, Pull Model</button>
        </div>
    </div>
  </div>

  <div class="toolbar">
    <h1>🍋 Lemonade Manager</h1>
    <div>
       Running Model: <code>{esc(loaded_model_name)}</code>
    </div>
    <form method="post" action="/unload" onsubmit="showLoading()">
      <button type="submit" class="btn-red">Unload ALL Models</button>
    </form>
  </div>

  <div style="overflow-x: auto;">
  <table>
    <thead>
      <tr>
        <th style="width: 20%;">Model ID</th>
        <th style="width: 15%;">Recipe / Backend</th>
        <th style="width: 5%; text-align:center;">DL</th>
        <th style="width: 10%;">Status</th>
        <th style="width: 50%;">Actions & Defaults</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  </div>

  <div class="section-container">
    <h2>⬇️ Pull New Model</h2>
    <form id="pull-form" onsubmit="showPullConfirm(event)">
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            
            <div class="form-group">
                <label class="form-label">Model Name (Namespace required)</label>
                <input type="text" id="pull_model_name" name="model_name" class="form-control" value="user." required>
            </div>
            
            <div class="form-group">
                <label class="form-label">Checkpoint (HuggingFace ID)</label>
                <input type="text" id="pull_checkpoint" name="checkpoint" class="form-control" placeholder="e.g. unsloth/Phi-4-mini-instruct-GGUF:Q4_K_M" required>
            </div>

            <div class="form-group">
                <label class="form-label">Recipe</label>
                <input type="text" id="pull_recipe" name="recipe" class="form-control" placeholder="e.g. llamacpp" required>
            </div>

            <div class="form-group">
                <label class="form-label">Multimodal Projector (Optional)</label>
                <div class="checkbox-wrapper">
                    <input type="checkbox" id="mmproj_enabled" onchange="toggleMmproj()">
                    <input type="text" id="mmproj_input" name="mmproj" class="form-control" placeholder="mmproj file path" disabled>
                </div>
            </div>
        </div>
        <div style="margin-top: 1rem;">
            <button type="submit" class="btn-primary">Pull Model</button>
        </div>
    </form>
  </div>

  <p class="text-muted text-sm" style="margin-top:1rem;">
    Defaults file: <code>{esc(str(RECIPE_FILE))}</code>
  </p>

  {stats_html}
</body>
</html>
"""
    return HTMLResponse(body)

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

# =============================================================================
# ACTION HANDLERS
# =============================================================================

@app.post("/load")
async def load_model_custom(
    model_name: str = Form(...),
    ctx_size: Optional[int] = Form(None),
    llamacpp_args: Optional[str] = Form(None),
    llamacpp_backend: Optional[str] = Form(None),
):
    """Action: 'Load Custom'. Uses whatever values are currently in the inputs."""
    await do_load(model_name, ctx_size, llamacpp_args, llamacpp_backend)
    return RedirectResponse(url="/", status_code=303)

@app.post("/defaults/load")
async def load_model_defaults(
    model_name: str = Form(...),
    llamacpp_backend: Optional[str] = Form(None)
):
    """
    Action: 'Load (Default)'. 
    Reads ctx/args from recipe_options.json.
    Backend override is preserved if user typed it.
    """
    options = get_model_options(model_name)
    ctx_size = options.get("ctx_size")
    llamacpp_args = options.get("llamacpp_args")

    # User input backend takes priority over stored default
    final_backend = llamacpp_backend if (llamacpp_backend and llamacpp_backend.strip()) else options.get("llamacpp_backend")

    await do_load(model_name, ctx_size, llamacpp_args, final_backend)
    return RedirectResponse(url="/", status_code=303)

@app.post("/defaults/set")
async def set_defaults(
    model_name: str = Form(...),
    ctx_size: Optional[int] = Form(None),
    llamacpp_args: Optional[str] = Form(None),
    llamacpp_backend: Optional[str] = Form(None),
):
    """Action: 'Save Defaults'. Updates recipe_options.json."""
    set_model_options(model_name, ctx_size, llamacpp_args, llamacpp_backend)
    return RedirectResponse(url="/", status_code=303)

@app.post("/unload")
async def unload_all_models_action():
    async with httpx.AsyncClient(timeout=TIMEOUT_LIGHT) as client:
        r = await client.post(f"{LEMONADE_BASE}/api/v1/unload", headers=get_headers())
        r.raise_for_status()
    return RedirectResponse(url="/", status_code=303)

@app.post("/unload/model")
async def unload_one_model_action(model_name: str = Form(...)):
    await do_unload_model(model_name)
    return RedirectResponse(url="/", status_code=303)

@app.post("/delete_model")
async def delete_model_action(model_name: str = Form(...)):
    """Handles the model deletion request."""
    await do_delete_model(model_name)
    return RedirectResponse(url="/", status_code=303)

@app.post("/disable")
async def disable_model_action(
    model_name: str = Form(...),
    disabled: str = Form(...),  # "1" or "0"
):
    set_disabled(model_name, disabled == "1")
    return RedirectResponse(url="/", status_code=303)

@app.post("/pull/stream")
async def pull_model_stream(
    model_name: str = Form(...),
    checkpoint: str = Form(...),
    recipe: str = Form(...),
    mmproj: Optional[str] = Form(None)
):
    """
    Proxies the Pull request to Lemonade Server with stream=true.
    Returns an event stream to the frontend for the progress bar.
    """
    payload = {
        "model_name": model_name,
        "checkpoint": checkpoint,
        "recipe": recipe,
        "stream": True 
    }
    if mmproj and mmproj.strip():
        payload["mmproj"] = mmproj.strip()

    # Create a generator to stream bytes from the upstream server to the client
    async def event_generator() -> AsyncGenerator[bytes, None]:
        async with httpx.AsyncClient(timeout=TIMEOUT_PULL) as client:
            # Use build_request/send to handle the streaming response properly
            req = client.build_request(
                "POST", 
                f"{LEMONADE_BASE}/api/v1/pull", 
                json=payload,
                headers=get_headers()
            )
            try:
                r = await client.send(req, stream=True)
                r.raise_for_status()
                async for chunk in r.aiter_bytes():
                    yield chunk
            except httpx.HTTPStatusError as e:
                # If immediate error, yield it as a data event so frontend can see it
                err_msg = json.dumps({"error": f"Upstream Error: {e.response.text}"})
                yield f"data: {err_msg}\n\n".encode('utf-8')
            except Exception as e:
                err_msg = json.dumps({"error": str(e)})
                yield f"data: {err_msg}\n\n".encode('utf-8')

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print(f"Starting Lemonade Manager on {MANAGER_HOST}:{MANAGER_PORT}...")
    print(f"Server Target: {LEMONADE_BASE}")
    if LEMONADE_KEY:
        print("API Key:       [Enabled]")
    print(f"Recipe File:   {RECIPE_FILE}")
    print(f"Prefs File:    {PREFS_FILE}")

    uvicorn.run("lemonade_manager:app", host=MANAGER_HOST, port=MANAGER_PORT, reload=False)
