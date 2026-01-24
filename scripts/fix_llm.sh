#!/usr/bin/env bash
#
# Ollama LLM Repair and Model Management Script
# Helps fix "exit status 2" and model loading issues
#

set -e  # Exit on error

echo "================================================================================================"
echo "  Ollama LLM Repair Script"
echo "  Platform: Fedora 43 | Ryzen AI MAX-395"
echo "================================================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# Check if Ollama is running
# ============================================================================

echo ""
echo "🔍 Checking Ollama service status..."

if ! systemctl is-active --quiet ollama 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Ollama service is not running${NC}"
    echo ""
    echo "Starting Ollama service..."
    
    if command -v systemctl &> /dev/null; then
        sudo systemctl start ollama
        sleep 2
        
        if systemctl is-active --quiet ollama; then
            echo -e "${GREEN}✅ Ollama service started${NC}"
        else
            echo -e "${RED}❌ Failed to start Ollama service${NC}"
            echo "Try manually: sudo systemctl start ollama"
            exit 1
        fi
    else
        echo "Please start Ollama manually: ollama serve"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Ollama is running${NC}"
fi

# ============================================================================
# List available models
# ============================================================================

echo ""
echo "📋 Currently installed models:"
ollama list

# ============================================================================
# Model selection
# ============================================================================

echo ""
echo "================================================================================================"
echo "  Model Selection"
echo "================================================================================================"
echo ""
echo "Recommended models for this hardware:"
echo "  1. mistral-nemo     (12B - Recommended, balanced)"
echo "  2. llama3.2         (3B - Smaller, faster)"
echo "  3. qwen2.5:7b       (7B - Good quality)"
echo "  4. gemma2:9b        (9B - Google model)"
echo ""

read -p "Enter model name (default: mistral-nemo): " MODEL
MODEL=${MODEL:-mistral-nemo}

echo ""
echo "Selected model: ${MODEL}"

# ============================================================================
# Optional cleanup
# ============================================================================

echo ""
read -p "Do you want to remove and re-download ${MODEL}? (y/N): " CLEANUP
CLEANUP=${CLEANUP:-n}

if [[ "$CLEANUP" =~ ^[Yy]$ ]]; then
    echo ""
    echo "🗑️  Removing ${MODEL}..."
    
    if ollama rm "$MODEL" 2>/dev/null; then
        echo -e "${GREEN}✅ Model removed${NC}"
    else
        echo -e "${YELLOW}⚠️  Model not found or already removed${NC}"
    fi
fi

# ============================================================================
# Pull/Update model
# ============================================================================

echo ""
echo "================================================================================================"
echo "  Downloading/Updating Model: ${MODEL}"
echo "================================================================================================"
echo ""
echo "This may take several minutes depending on model size..."
echo ""

if ollama pull "$MODEL"; then
    echo ""
    echo -e "${GREEN}✅ Model ${MODEL} downloaded successfully${NC}"
else
    echo ""
    echo -e "${RED}❌ Failed to download model${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check internet connection"
    echo "  2. Verify model name: ollama list"
    echo "  3. Check disk space: df -h"
    echo "  4. View Ollama logs: journalctl -u ollama -n 50"
    exit 1
fi

# ============================================================================
# Test the model
# ============================================================================

echo ""
echo "================================================================================================"
echo "  Testing Model"
echo "================================================================================================"
echo ""
echo "Running quick test query..."
echo ""

TEST_RESPONSE=$(ollama run "$MODEL" "Respond with exactly: 'Model is working correctly.'" 2>&1)

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✅ Model test successful!${NC}"
    echo ""
    echo "Response:"
    echo "$TEST_RESPONSE"
else
    echo -e "${RED}❌ Model test failed${NC}"
    echo ""
    echo "Error output:"
    echo "$TEST_RESPONSE"
    echo ""
    echo "Check Ollama logs: journalctl -u ollama -n 50"
    exit 1
fi

# ============================================================================
# Update .env file
# ============================================================================

echo ""
echo "================================================================================================"
echo "  Updating Configuration"
echo "================================================================================================"

ENV_FILE=".env"

if [ ! -f "$ENV_FILE" ]; then
    echo ""
    echo "Creating .env file..."
    
    if [ -f ".env.template" ]; then
        cp .env.template .env
        echo "✅ Created .env from template"
    else
        touch .env
        echo "✅ Created empty .env"
    fi
fi

# Update or add LLM_MODEL
if grep -q "^LLM_MODEL=" "$ENV_FILE"; then
    # Update existing
    sed -i "s/^LLM_MODEL=.*/LLM_MODEL=${MODEL}/" "$ENV_FILE"
    echo "✅ Updated LLM_MODEL in .env"
else
    # Add new
    echo "LLM_MODEL=${MODEL}" >> "$ENV_FILE"
    echo "✅ Added LLM_MODEL to .env"
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "================================================================================================"
echo "  ✅ Repair Complete!"
echo "================================================================================================"
echo ""
echo "Configuration:"
echo "  Model: ${MODEL}"
echo "  Config: .env"
echo ""
echo "Next steps:"
echo "  1. Restart your application: streamlit run src/app.py"
echo "  2. If issues persist, check logs: journalctl -u ollama -f"
echo "  3. Monitor Ollama: htop (look for 'ollama' process)"
echo ""
echo "Environment variables set:"
echo "  LLM_MODEL=${MODEL}"
echo ""
echo "To use a different model, run this script again or edit .env manually."
echo ""
echo "================================================================================================"

exit 0
