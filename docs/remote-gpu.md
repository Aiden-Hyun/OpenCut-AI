# Remote GPU for tts-service and inpaint-service

The two compute-heavy services — tts-service (XTTS voice cloning) and
inpaint-service (STTN subtitle removal) — run CPU-only torch in the local
stack. On an NVIDIA GPU the same pinned torch versions run roughly:

- XTTS synthesis: ~5-10x faster
- STTN inpainting: ~10-50x faster

This runbook rents a GPU box, runs just those two services on it with CUDA
wheels, and flips the local ai-backend to use them over an SSH tunnel.
Everything else (db, web, ai-backend, whisper, ...) stays on your Mac.

## Security model

The services have **no authentication**. The GPU-host compose file therefore
binds their ports to `127.0.0.1` only — nothing is reachable from the
internet. The only way in is the SSH tunnel below. Do not change the port
bindings to `0.0.0.0` and do not open 8422/8427 in the host firewall.

## Cost

Any on-demand mid-tier GPU works (RTX 3090/4090, A4000/A5000, L4):
roughly $0.2-0.4/hr on RunPod, Lambda, Vast.ai, etc. XTTS needs ~4 GB
VRAM; STTN is light (~2 GB). Stop the box when you are done — the model
caches live in named volumes, so restarts are cheap but a deleted pod is
not (weights re-download on first use, ~2 GB for XTTS, ~66 MB for STTN).

## 1. Set up the GPU host (once)

Rent an Ubuntu box with an NVIDIA GPU (RunPod / Lambda / any provider) and
SSH access. On the box:

```bash
# Docker + NVIDIA container toolkit (skip what the image already ships)
curl -fsSL https://get.docker.com | sh
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

git clone https://github.com/Aiden-Hyun/OpenCut-AI.git
cd OpenCut-AI
git checkout localization-pipeline
```

## 2. Build and start the services

```bash
docker compose -f docker-compose.gpu-host.yml up -d --build
```

This builds both Dockerfiles with `TORCH_VARIANT=cu121`, which installs the
lockfile as usual and then swaps torch (and torchaudio for tts) for the
CUDA 12.1 wheels at the same pinned versions. First build downloads ~3 GB
of CUDA wheels.

Verify the GPU is visible in-container and the services are healthy:

```bash
docker compose -f docker-compose.gpu-host.yml exec tts-service nvidia-smi
curl -s http://127.0.0.1:8422/health
curl -s http://127.0.0.1:8427/health
```

Both apps auto-select the device at runtime (`torch.cuda.is_available()`),
so the same image also runs on CPU. After the first TTS/inpaint request,
`/health` (inpaint) and `/models` (tts) report `"device": "cuda"`.

## 3. Tunnel + flip on the Mac

In one terminal (stays in the foreground, auto-reconnects):

```bash
scripts/remote-gpu.sh tunnel user@gpu-host
```

This forwards local ports 18422/18427 to the GPU host's 8422/8427. The
1-prefixed ports avoid clashing with the local containers, which can keep
running.

In another terminal:

```bash
scripts/remote-gpu.sh on       # recreate ai-backend pointing at the tunnel
scripts/remote-gpu.sh status   # remote/local reachability + current mode
```

`on` layers `docker-compose.remote-gpu.yml` over the usual local invocation
and recreates only ai-backend with:

- `OPENCUTAI_TTS_SERVICE_URL=http://host.docker.internal:18422`
- `OPENCUTAI_INPAINT_SERVICE_URL=http://host.docker.internal:18427`

## 4. Flip back

```bash
scripts/remote-gpu.sh off      # ai-backend -> local containers again
```

Then ctrl-c the tunnel and stop (or delete) the GPU box.

## Troubleshooting

- `status` shows the remote services unreachable: the tunnel is not up, or
  the services are still building/downloading models on the GPU host
  (`docker compose -f docker-compose.gpu-host.yml logs -f`).
- `could not select device driver "nvidia"`: nvidia-container-toolkit is
  missing or Docker was not restarted after `nvidia-ctk runtime configure`.
- TTS/inpaint still slow after flipping: check `remote-gpu.sh status` says
  mode REMOTE, and check the remote `/health` reports `"device": "cuda"`
  after the first job (a CPU fallback means the container cannot see the
  GPU — check `nvidia-smi` in-container).
