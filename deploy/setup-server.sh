#!/usr/bin/env bash
set -euo pipefail

echo "=== Legal AI Platform — Server Setup ==="

# 1. Install Docker
if ! command -v docker &>/dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    echo "Docker installed. Re-login or run: newgrp docker"
fi

# 2. Install Docker Compose plugin
if ! docker compose version &>/dev/null; then
    echo "Installing Docker Compose plugin..."
    sudo apt-get update && sudo apt-get install -y docker-compose-plugin
fi

# 3. Install Nginx
if ! command -v nginx &>/dev/null; then
    echo "Installing Nginx..."
    sudo apt-get update && sudo apt-get install -y nginx
fi

# 4. Copy nginx config
sudo cp deploy/nginx.conf /etc/nginx/sites-available/legal-ai
sudo ln -sf /etc/nginx/sites-available/legal-ai /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

# 5. Check .env
if [ ! -f .env ]; then
    echo "ERROR: .env not found. Copy .env.example and fill in API keys:"
    echo "  cp .env.example .env"
    exit 1
fi

# 6. Start services
echo "Starting services..."
docker compose up -d

echo ""
echo "=== Done ==="
echo "Open WebUI: http://$(hostname -I | awk '{print $1}'):3000"
echo "Nginx proxy: http://$(hostname -I | awk '{print $1}')"
