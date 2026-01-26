#!/bin/bash
# ==============================================================================
# Garden Ollama Turbo Mode Restart Script
# ==============================================================================
# 
# Purpose: Restart the Garden Podman container with optimized Ollama settings
#          for high-performance inference on AMD Ryzen AI MAX-395.
#
# Optimizations:
# - OLLAMA_NUM_PARALLEL=4    : Process up to 4 concurrent requests
# - OLLAMA_MAX_LOADED_MODELS=2 : Keep 2 models in memory simultaneously
# - HSA_OVERRIDE flags       : ROCm GPU acceleration for Radeon 8060S iGPU
#
# Hardware: AMD Ryzen AI MAX-395, 128GB RAM, Radeon 8060S (32GB VRAM)
# ==============================================================================

set -euo pipefail

# Configuration
CONTAINER_NAME="${GARDEN_CONTAINER_NAME:-garden-production}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# Functions
# ==============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_podman() {
    if ! command -v podman &> /dev/null; then
        log_error "Podman is not installed. Please install podman first."
        exit 1
    fi
    log_success "Podman detected: $(podman --version)"
}

stop_existing_container() {
    log_info "Checking for existing container: $CONTAINER_NAME"
    
    if podman ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        log_warning "Found existing container '$CONTAINER_NAME'. Stopping and removing..."
        podman stop "$CONTAINER_NAME" 2>/dev/null || true
        podman rm "$CONTAINER_NAME" 2>/dev/null || true
        log_success "Container removed"
    else
        log_info "No existing container found"
    fi
}

start_turbo_container() {
    log_info "Starting Garden Ollama in TURBO MODE..."
    
    # WARNING: User should verify their volume mappings and model paths
    log_warning "This script uses default volume mappings. Please verify:"
    log_warning "  - Ollama models path: /path/to/your/ollama/models"
    log_warning "  - Data path: /path/to/your/garden/data"
    log_warning "Edit this script to customize volume mappings if needed."
    
    echo ""
    read -p "Continue with default settings? (y/N): " -n 1 -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cancelled by user. Please edit this script with your volume paths."
        exit 0
    fi
    
    # Start container with turbo settings
    # NOTE: User should customize volume paths as needed
    podman run -d \
        --name "$CONTAINER_NAME" \
        --restart unless-stopped \
        -p "${OLLAMA_PORT}:11434" \
        -v ollama-models:/root/.ollama \
        -e OLLAMA_HOST="$OLLAMA_HOST:11434" \
        -e OLLAMA_NUM_PARALLEL=4 \
        -e OLLAMA_MAX_LOADED_MODELS=2 \
        -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
        -e HSA_ENABLE_SDMA=0 \
        --device /dev/kfd \
        --device /dev/dri \
        --security-opt seccomp=unconfined \
        --group-add video \
        ollama/ollama:latest
    
    log_success "Container started with TURBO settings!"
}

verify_container() {
    log_info "Verifying container status..."
    sleep 3
    
    if podman ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        log_success "Container is running!"
        
        # Show container info
        echo ""
        log_info "Container details:"
        podman inspect "$CONTAINER_NAME" --format "
  - Name: {{.Name}}
  - Status: {{.State.Status}}
  - Ports: {{range .NetworkSettings.Ports}}{{.}}{{end}}
  - Created: {{.Created}}"
        
        # Wait for Ollama API to be ready
        log_info "Waiting for Ollama API to be ready..."
        for i in {1..30}; do
            if curl -s "http://localhost:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1; then
                log_success "Ollama API is ready!"
                return 0
            fi
            sleep 1
        done
        
        log_warning "Ollama API did not respond within 30 seconds"
        log_info "You can check logs with: podman logs $CONTAINER_NAME"
    else
        log_error "Container failed to start. Check logs with: podman logs $CONTAINER_NAME"
        exit 1
    fi
}

show_models() {
    log_info "Fetching available models..."
    echo ""
    
    if curl -s "http://localhost:${OLLAMA_PORT}/api/tags" | jq -r '.models[] | "  - \(.name) (\(.size / 1024 / 1024 / 1024 | floor)GB)"' 2>/dev/null; then
        echo ""
        log_success "Models listed successfully"
    else
        log_warning "Could not fetch models. The API might still be initializing."
        log_info "Check models later with: curl http://localhost:${OLLAMA_PORT}/api/tags | jq"
    fi
}

show_turbo_config() {
    echo ""
    log_info "========================================="
    log_info "TURBO MODE Configuration"
    log_info "========================================="
    echo "  OLLAMA_NUM_PARALLEL=4       (4 concurrent requests)"
    echo "  OLLAMA_MAX_LOADED_MODELS=2  (2 models in memory)"
    echo "  HSA_OVERRIDE_GFX_VERSION    (ROCm GPU acceleration)"
    echo "  Port: ${OLLAMA_PORT}"
    log_info "========================================="
    echo ""
}

# ==============================================================================
# Main Script
# ==============================================================================

main() {
    echo ""
    log_info "=========================================="
    log_info "  Garden Ollama TURBO MODE Restart"
    log_info "=========================================="
    echo ""
    
    check_podman
    stop_existing_container
    start_turbo_container
    verify_container
    show_turbo_config
    show_models
    
    echo ""
    log_success "Garden Ollama is running in TURBO MODE! 🚀"
    echo ""
    log_info "Test the API with:"
    echo "  curl http://localhost:${OLLAMA_PORT}/api/tags"
    echo ""
    log_info "View logs with:"
    echo "  podman logs -f $CONTAINER_NAME"
    echo ""
}

# Run main
main
