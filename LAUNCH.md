# Launching the Call Center Assistant

## Architecture

- **GPU node (AWS ParallelCluster)**: runs the RAG pipeline (LaBSE retriever + Salamandra LLM) behind a FastAPI server on port `8000`, and the Gradio UI on port `19000`.
- **Local Mac**: opens an SSH tunnel to forward `localhost:19000` to the GPU node, then accesses the app via browser.

---

## Step 1 — Connect to the GPU node (on your Mac)

Open a terminal and connect to the cluster as usual:

```bash
./vscode-connect.sh
ssh pcluster-ssh
```

From the head node, start an interactive GPU job (or attach to an existing one) and navigate to the project:

```bash
cd /workspace/NLU/cvlachos/Pilot_3
```

---

## Step 2 — Start the FastAPI server (on the GPU node)

Open a tmux session and start the pipeline API in the first pane:

```bash
tmux
uvicorn server:app --host 0.0.0.0 --port 8000
```

> **Scrolling in tmux:** Enter copy mode with `Ctrl+b [`, then scroll with arrow keys or `PgUp`/`PgDn`. Press `q` to exit.
> To enable mouse scrolling permanently, add `set -g mouse on` to `~/.tmux.conf`, or run it once with `Ctrl+b :set -g mouse on Enter`.

---

## Step 3 — Start the Gradio UI (on the GPU node)

Split the tmux pane (`Ctrl+b` then `%`) and run:

```bash
python app.py
```

You should see:
```
* Running on local URL: http://0.0.0.0:19000
```

---

## Step 4 — Open the SSH tunnel (on your Mac)

Open a new terminal on your Mac. Run `vscode-connect.sh` to push a fresh temporary key, then immediately open the tunnel:

```bash
./vscode-connect.sh
ssh -L 19000:gpu-1-spot-dy-compute-g6-2xlarge-9:19000 -N pcluster-ssh
```

The terminal will appear to hang — that is expected. Keep it open for as long as you use the app.

> If you get `Address already in use`, free the port first:
> ```bash
> lsof -ti:19000 | xargs kill -9
> ```

---

## Step 5 — Open the app

Open your browser and go to:

```
http://localhost:19000
```

---

## Shutting down

1. In the tunnel terminal on your Mac: `Ctrl+C`
2. In each tmux pane on the GPU node: `Ctrl+C`


## Interesting cases to check

Test case ids: 20, 212, 213