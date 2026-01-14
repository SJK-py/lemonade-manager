# Lemonade Manager 🍋

**Version:** v2.2

A lightweight, single-file Web UI manager for [Lemonade Server](https://github.com/lemonade-sdk/lemonade).

This tool provides a clean interface to download, load, unload, delete, and configure models running on your local Lemonade instance. It is designed to run alongside the Lemonade server and persists configuration preferences directly to your Lemonade recipe path.

## Screenshot
![lemonade_manager_screenshot](https://github.com/user-attachments/assets/8211dd49-58dd-4044-9700-8a8b028c0e19)
*(Note: UI may vary slightly in newer versions)*

## Features

* **Model Management:** View all available models, their running status, and recipe details.
* **Pull / Download Models:** dedicated interface to pull new models directly from HuggingFace (via Lemonade) with real-time streaming progress logs.
* **Load/Unload Controls:** Quickly start or stop models. Supports custom backend overrides (e.g., swapping `vulkan` vs `cpu` on the fly).
* **Delete Models:** Remove model files from disk directly from the UI.
* **Persistent Configuration:** Save default context sizes (`ctx_size`), `llamacpp_args`, and backend preferences to `recipe_options.json`.
* **Hiding Models:** "Disable" models to hide them from the main view without deleting the files (stored locally in `manager_prefs.json`).
* **Authentication Support:** Optional Bearer token support (`LEMONADE_KEY`) for securing connections to the Lemonade Server.
* **Responsive UI:** Dark-mode enabled, lightweight HTML/CSS interface with zero external frontend dependencies.

## Prerequisites

* Python 3.8+
* A running instance of **Lemonade Server** (reachable via HTTP).

## Installation (Virtual Environment)

1. **Clone or download** the `lemonade_manager.py` and `requirements.txt` file to your desired directory (e.g., `/opt/lemonade-manager`).
2. **Create a virtual environment** to keep dependencies isolated:
   ```bash
   python3 -m venv venv
   ```
3. **Activate the environment**:
   ```bash
   source venv/bin/activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Manual Start**

Ensure your virtual environment is active, then run:

```bash
python lemonade_manager.py
```

By default, the UI will be accessible at http://localhost:9000.

**Environment Variables**

You can override defaults by setting environment variables before running the script:
| Variable | Default | Description |
|---|---|---|
| LEMONADE_BASE | http://localhost:8000 | Base URL of the Lemonade Server API. |
| LEMONADE_KEY | (empty) | API Key (Bearer Token) if your Lemonade Server requires auth. |
| MANAGER_HOST | 0.0.0.0 | Interface to bind the Manager UI to. |
| MANAGER_PORT | 9000 | Port to run this Web UI on. |
| RECIPE_FILE | ~/.cache/lemonade/recipe_options.json | Path to the shared Lemonade server config file. |
| PREFS_FILE | manager_prefs.json | Path to store local UI preferences (hidden models list). |
| TIMEOUT_LOAD | 120.0 | Timeout (seconds) for loading models. |
| TIMEOUT_PULL | 3600.0 | Timeout (seconds) for downloading large models. |
| TIMEOUT_LIGHT | 10.0 | Timeout (seconds) for light ops (stats, health checks). |

**Pulling New Models**
The Manager includes a "Pull New Model" section at the bottom of the UI.
* Model Name: The identifier for the model (e.g., user.mistral-7b).
* Checkpoint: The HuggingFace ID (e.g., unsloth/Phi-4-mini-instruct-GGUF:Q4_K_M).
* Recipe: The backend recipe to use (usually llamacpp).
* Multimodal: Optional path to an mmproj file if required.
Note: The download process streams logs to the browser window. Do not close the tab until the download is complete.

**Running as a System Service (Systemd)**

For a persistent setup, use systemd to auto-start the manager.
1. Create the service file:
```bash
sudo nano /etc/systemd/system/lemonade-manager.service
```

2. Paste the following configuration:
Replace /opt/lemonade-manager with your actual installation path and your_username with your Linux user.
```ini
[Unit]
Description=Lemonade Manager Web UI
After=network.target

[Service]
Type=simple
User=your_username
Group=your_username

# Set the working directory to where the script is located
WorkingDirectory=/opt/lemonade-manager

# Use the python executable INSIDE the venv
ExecStart=/opt/lemonade-manager/venv/bin/python lemonade_manager.py

# Restart automatically if it crashes
Restart=on-failure
RestartSec=5

# Environment Overrides
Environment=LEMONADE_BASE=http://localhost:8000
Environment=MANAGER_PORT=9000
# Environment=LEMONADE_KEY=your_secret_key
# Environment=TIMEOUT_PULL=3600

[Install]
WantedBy=multi-user.target
```

3. Enable and Start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable lemonade-manager
sudo systemctl start lemonade-manager
```

4. Check Status:
```bash
sudo systemctl status lemonade-manager
```

## License
MIT
