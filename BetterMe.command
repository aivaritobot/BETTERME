#!/bin/bash
# Doble clic en este archivo desde Finder para abrir BETTERME Desktop.
cd "$(dirname "$0")"
python3 -m app.main
