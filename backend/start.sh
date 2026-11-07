#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
python -c 'import sys; assert sys.version_info[:2] == (3, 10) and sys.version_info >= (3, 10, 8), "T-Scanner requires Python 3.10.8 or later Python 3.10 patch releases"'
exec python main.py
