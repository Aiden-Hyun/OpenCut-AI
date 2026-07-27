#!/usr/bin/env bash
# Flip the local OpenCut stack between local (CPU) and remote (GPU)
# tts-service / inpaint-service. See docs/remote-gpu.md for the runbook.
#
#   remote-gpu.sh tunnel <ssh-host>  # SSH tunnel to the GPU host (foreground)
#   remote-gpu.sh on                 # ai-backend -> remote services via tunnel
#   remote-gpu.sh off                # ai-backend -> local containers again
#   remote-gpu.sh status             # tunnel reachability + current mode
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Remote services bind to 127.0.0.1:<port> on the GPU host (no auth on the
# services, so nothing is exposed publicly). The tunnel forwards them to
# 1-prefixed local ports so local tts/inpaint containers can keep running.
TTS_REMOTE_PORT=8422
INPAINT_REMOTE_PORT=8427
TTS_TUNNEL_PORT=18422
INPAINT_TUNNEL_PORT=18427

COMPOSE_LOCAL=(docker compose -f docker-compose.yml -f docker-compose.local.yml)
COMPOSE_REMOTE=("${COMPOSE_LOCAL[@]}" -f docker-compose.remote-gpu.yml)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

usage() {
    cat <<'EOF'
Flip the local OpenCut stack between local (CPU) and remote (GPU)
tts-service / inpaint-service. See docs/remote-gpu.md for the runbook.

  remote-gpu.sh tunnel <ssh-host>  # SSH tunnel to the GPU host (foreground)
  remote-gpu.sh on                 # ai-backend -> remote services via tunnel
  remote-gpu.sh off                # ai-backend -> local containers again
  remote-gpu.sh status             # tunnel reachability + current mode
EOF
    exit 1
}

cmd_tunnel() {
    local host="${1:-}"
    if [ -z "$host" ]; then
        log_error "usage: remote-gpu.sh tunnel <ssh-host>"
        exit 1
    fi
    # Poor man's autossh: plain ssh dies on network blips, so loop and
    # reconnect. If you have autossh installed, the equivalent is:
    #   autossh -M 0 -N -L 18422:127.0.0.1:8422 -L 18427:127.0.0.1:8427 <host>
    log_info "Tunnelling ${TTS_TUNNEL_PORT}->:${TTS_REMOTE_PORT} and ${INPAINT_TUNNEL_PORT}->:${INPAINT_REMOTE_PORT} via ${host} (ctrl-c to stop)"
    while true; do
        ssh -N \
            -o ServerAliveInterval=15 \
            -o ServerAliveCountMax=3 \
            -o ExitOnForwardFailure=yes \
            -L "${TTS_TUNNEL_PORT}:127.0.0.1:${TTS_REMOTE_PORT}" \
            -L "${INPAINT_TUNNEL_PORT}:127.0.0.1:${INPAINT_REMOTE_PORT}" \
            "$host" || true
        log_warn "Tunnel disconnected; reconnecting in 3s (ctrl-c to stop)..."
        sleep 3
    done
}

cmd_on() {
    cd "$PROJECT_ROOT"
    log_info "Recreating ai-backend with remote GPU service URLs..."
    "${COMPOSE_REMOTE[@]}" up -d --no-deps ai-backend
    log_info "ai-backend now targets the remote services through the tunnel."
    log_info "Keep 'remote-gpu.sh tunnel <ssh-host>' running in another terminal."
}

cmd_off() {
    cd "$PROJECT_ROOT"
    log_info "Recreating ai-backend with local service URLs..."
    "${COMPOSE_LOCAL[@]}" up -d --no-deps ai-backend
    log_info "ai-backend now targets the local tts/inpaint containers."
}

check_health() {
    # check_health <label> <url>
    if curl -fsS -m 3 "$2" >/dev/null 2>&1; then
        echo -e "  $1: ${GREEN}reachable${NC}"
    else
        echo -e "  $1: ${RED}unreachable${NC}"
    fi
}

cmd_status() {
    cd "$PROJECT_ROOT"
    echo "Remote services (via SSH tunnel):"
    check_health "tts-service     (127.0.0.1:${TTS_TUNNEL_PORT})" "http://127.0.0.1:${TTS_TUNNEL_PORT}/health"
    check_health "inpaint-service (127.0.0.1:${INPAINT_TUNNEL_PORT})" "http://127.0.0.1:${INPAINT_TUNNEL_PORT}/health"
    echo "Local services:"
    check_health "tts-service     (127.0.0.1:${TTS_REMOTE_PORT})" "http://127.0.0.1:${TTS_REMOTE_PORT}/health"
    check_health "inpaint-service (127.0.0.1:${INPAINT_REMOTE_PORT})" "http://127.0.0.1:${INPAINT_REMOTE_PORT}/health"

    local cid
    cid="$("${COMPOSE_LOCAL[@]}" ps -q ai-backend 2>/dev/null || true)"
    if [ -z "$cid" ]; then
        log_warn "ai-backend is not running."
    elif docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$cid" \
            | grep -q "host.docker.internal:${TTS_TUNNEL_PORT}"; then
        log_info "ai-backend mode: REMOTE (using the tunnel)"
    else
        log_info "ai-backend mode: LOCAL (using local containers)"
    fi
}

case "${1:-}" in
    tunnel) shift; cmd_tunnel "$@" ;;
    on)     cmd_on ;;
    off)    cmd_off ;;
    status) cmd_status ;;
    *)      usage ;;
esac
