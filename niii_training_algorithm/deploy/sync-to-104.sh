#!/bin/bash
set -euo pipefail
HOST=10.65.48.104
USER=root
PASS='Zq@163.com'
REMOTE=/niii_machine_version/AI_trainning_platform/boot-vison-python-master
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Sync algorithm code to 104..."
sshpass -p "$PASS" rsync -avz --delete \
  --exclude 'mydjango_project/venv/' \
  --exclude 'venv/' \
  --exclude '__pycache__/' \
  --exclude '.git/' \
  --exclude 'Miniconda3*.sh' \
  --exclude 'ML_backend.zip' \
  --include 'resources/' \
  --include 'resources/*.pt' \
  --exclude '*.pt' \
  --exclude 'runs/' \
  --exclude 'algorithm_model/SAM/pretrained_checkpoint/' \
  --exclude 'algorithm_model/SAM/demo/' \
  --exclude 'db.sqlite3' \
  "$ROOT/" "${USER}@${HOST}:${REMOTE}/"

echo "Restart django on 104..."
sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "${USER}@${HOST}" \
  'pkill -f "manage.py runserver 0.0.0.0:8008" 2>/dev/null || true; sleep 1; cd /niii_machine_version/AI_trainning_platform/boot-vison-python-master && nohup mydjango_project/venv/bin/python manage.py runserver 0.0.0.0:8008 > /project/niii/boot-vision/backend/django_console.log 2>&1 &'

echo "Done."
