#Requires -Version 5.1
param(
    [string]$HostName = "10.65.48.104",
    [string]$User = "root",
    [string]$Password = "Zq@163.com",
    [string]$Remote = "/niii_machine_version/AI_trainning_platform/boot-vison-python-master",
    [switch]$RestartDjango = $true
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RestartFlag = if ($RestartDjango) { "1" } else { "0" }

$env:NIII_ALGO_SYNC_ROOT = $Root
$env:NIII_ALGO_SYNC_REMOTE = $Remote
$env:NIII_ALGO_SYNC_HOST = $HostName
$env:NIII_ALGO_SYNC_USER = $User
$env:NIII_ALGO_SYNC_PASSWORD = $Password
$env:NIII_ALGO_SYNC_RESTART = $RestartFlag

$pyScript = @'
import os, tarfile, tempfile, paramiko, sys
root = os.environ["NIII_ALGO_SYNC_ROOT"]
remote = os.environ["NIII_ALGO_SYNC_REMOTE"]
host = os.environ["NIII_ALGO_SYNC_HOST"]
user = os.environ["NIII_ALGO_SYNC_USER"]
password = os.environ["NIII_ALGO_SYNC_PASSWORD"]
restart = os.environ.get("NIII_ALGO_SYNC_RESTART") == "1"

def skip(rel):
    rel = rel.replace("\\\\", "/")
    for p in ("mydjango_project/venv/", "venv/", ".git/", "runs/",
              "algorithm_model/SAM/pretrained_checkpoint/", "algorithm_model/SAM/demo/"):
        if rel.startswith(p):
            return True
    if "__pycache__" in rel.split("/"):
        return True
    base = os.path.basename(rel)
    if base.endswith(".pt") and not rel.replace("\\\\", "/").startswith("resources/"):
        return True
    if base.startswith("Miniconda3") and base.endswith(".sh"):
        return True
    return False

tar_path = tempfile.mktemp(suffix=".tar.gz")
with tarfile.open(tar_path, "w:gz") as tar:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "venv", "__pycache__"}]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            if skip(rel):
                continue
            tar.add(full, arcname=rel.replace("\\\\", "/"))
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=user, password=password, timeout=30)
sftp = client.open_sftp()
sftp.put(tar_path, "/tmp/niii-algorithm-sync.tar.gz")
sftp.close()
os.remove(tar_path)
_, stdout, stderr = client.exec_command(
    f"mkdir -p {remote} && tar -xzf /tmp/niii-algorithm-sync.tar.gz -C {remote} && echo SYNC_OK"
)
out = stdout.read().decode()
err = stderr.read().decode()
code = stdout.channel.recv_exit_status()
print(out, end="")
if err.strip():
    print(err, file=sys.stderr)
if code != 0:
    sys.exit(code)
if restart:
    client.exec_command(
        'pkill -f "manage.py runserver 0.0.0.0:8008" 2>/dev/null || true; '
        'sleep 1; cd /niii_machine_version/AI_trainning_platform/boot-vison-python-master && '
        'nohup mydjango_project/venv/bin/python manage.py runserver 0.0.0.0:8008 '
        '> /project/niii/boot-vision/backend/django_console.log 2>&1 &'
    )
client.close()
print("DONE")
'@

$pyScript | python -
if ($LASTEXITCODE -ne 0) { throw "sync failed" }
Write-Host "Sync complete." -ForegroundColor Green
