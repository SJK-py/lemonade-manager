# Lemonade Manager 🍋

A lightweight, single-file Web UI manager for [Lemonade Server](https://github.com/lemonade-sdk/lemonade).

This tool provides a clean interface to load, unload, and configure models running on your local Lemonade instance. It is designed to run alongside the Lemonade server and persists configuration preferences directly to your Lemonade recipe path.

## Screenshot
![lemonade_manager_screenshot](https://github.com/user-attachments/assets/8211dd49-58dd-4044-9700-8a8b028c0e19)

## Features

* **Model Management:** View all available models, their load status, and download status.
* **Load/Unload Controls:** Quickly start or stop models with a single click.
* **Persistent Configuration:** Save default context sizes (`ctx_size`), `llamacpp_args`, and backend preferences (e.g., `vulkan`, `cuda`) to `recipe_options.json`.
* **Backend Overrides:** dedicated input for specifying `llamacpp_backend` (useful for testing Vulkan vs CPU offloading on the fly).
* **Hiding Models:** "Disable" models to hide them from the main view without deleting the files (stored locally in `manager_prefs.json`).
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
| MANAGER_PORT | 9000 | Port to run this Web UI on. |
| RECIPE_FILE | ~/.cache/lemonade/recipe_options.json | Path to the shared Lemonade server config file. |
| PREFS_FILE | manager_prefs.json | Path to store local UI preferences (hidden models list). |
| TIMEOUT_LOAD | 120.0 | Timeout (seconds) for loading heavy models. |
| TIMEOUT_LIGHT | 10.0 | Timeout (seconds) for light operations like stats or health checks. |

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

# Optional: Override environment variables here
# Environment=LEMONADE_BASE=http://localhost:8000
# Environment=MANAGER_PORT=9000

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
