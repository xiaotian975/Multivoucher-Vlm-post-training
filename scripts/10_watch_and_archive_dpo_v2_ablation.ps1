param(
    [string]$HostName = "connect.westc.seetacloud.com",
    [int]$Port = 51327,
    [string]$User = "root",
    [string]$RemoteProject = "/root/autodl-tmp/VLM-Post-Training",
    [string]$LocalRepo = "D:\Reserach\Projects\VLM post-training",
    [string]$RunBase = "outputs/runtime/dpo_v2_ablation_5gpu",
    [int]$PollSeconds = 300,
    [switch]$ShutdownOnSuccess,
    [switch]$RemoveSelfOnSuccess
)

$ErrorActionPreference = "Stop"

function Invoke-Remote {
    param([string]$Command)
    $target = $User + "@" + $HostName
    $output = ssh -p $Port $target "cd '$RemoteProject' && $Command"
    if ($null -eq $output) {
        return ""
    }
    return ($output -join "`n")
}

function Copy-RemoteFile {
    param([string]$RemotePath, [string]$LocalPath)
    $targetPath = $User + "@" + $HostName + ":" + $RemoteProject + "/" + $RemotePath
    scp -P $Port $targetPath $LocalPath
}

Set-Location $LocalRepo
$localLogRoot = Join-Path $LocalRepo "outputs\runtime\local_watchers"
New-Item -ItemType Directory -Force -Path $localLogRoot | Out-Null
$watchLog = Join-Path $localLogRoot ("dpo_v2_ablation_watch_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

"watch_start=$(Get-Date -Format o)" | Tee-Object -FilePath $watchLog -Append | Out-Null

while ($true) {
    try {
        $runId = (Invoke-Remote "cat '$RunBase/LATEST_RUN_ID' 2>/dev/null || true").Trim()
        if (-not $runId) {
            "no_run_id_yet=$(Get-Date -Format o)" | Tee-Object -FilePath $watchLog -Append | Out-Null
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        $runRoot = "$RunBase/$runId"
        $failed = (Invoke-Remote "test -f '$runRoot/FAILED' && cat '$runRoot/FAILED' || true").Trim()
        if ($failed) {
            "remote_failed=$runRoot" | Tee-Object -FilePath $watchLog -Append | Out-Null
            $failed | Tee-Object -FilePath $watchLog -Append | Out-Null
            exit 2
        }
        $ready = (Invoke-Remote "test -f '$runRoot/READY_TO_ARCHIVE' && cat '$runRoot/READY_TO_ARCHIVE' || true").Trim()
        if (-not $ready) {
            "not_ready run_id=$runId time=$(Get-Date -Format o)" | Tee-Object -FilePath $watchLog -Append | Out-Null
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        "ready run_id=$runId" | Tee-Object -FilePath $watchLog -Append | Out-Null
        $archiveRel = (Invoke-Remote "cat '$runRoot/archive_tar_path'").Trim()
        $tmpArchive = Join-Path $env:TEMP ("phase08_loss_ablation_{0}.tar.gz" -f $runId)
        Copy-RemoteFile $archiveRel $tmpArchive
        tar -xzf $tmpArchive -C $LocalRepo

        $archiveDir = Join-Path $LocalRepo ("docs\experiments\phase08_loss_ablation_{0}" -f $runId)
        $appendPath = Join-Path $archiveDir "README_APPEND.md"
        if (-not (Test-Path $appendPath)) {
            throw "README append file missing: $appendPath"
        }
        $marker = "phase08_loss_ablation_$runId"
        $readme = Join-Path $LocalRepo "README.md"
        $readmeText = Get-Content -Encoding UTF8 $readme -Raw
        if ($readmeText -notmatch [regex]::Escape($marker)) {
            Add-Content -Encoding UTF8 -Path $readme -Value (Get-Content -Encoding UTF8 $appendPath -Raw)
        }

        $manifestCheck = @'
import hashlib
import json
import sys
from pathlib import Path

archive = Path(sys.argv[1])
manifest = json.loads((archive / "artifact_manifest.json").read_text(encoding="utf-8"))
bad = []
for row in manifest["files"]:
    path = archive / row["path"]
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if h != row["sha256"]:
        bad.append(row["path"])
if bad:
    raise SystemExit(f"sha256_mismatch={bad[:5]}")
print(f"manifest_ok files={len(manifest['files'])}")
'@
        python -c $manifestCheck $archiveDir

        rg -n "phase08_loss_ablation_|High-risk Miss Rate|Audit Accuracy|AuxDPO|IPO" README.md $archiveDir | Out-Null
        git diff --check -- README.md "docs/experiments/phase08_loss_ablation_$runId"

        "archive_ok run_id=$runId time=$(Get-Date -Format o)" | Tee-Object -FilePath $watchLog -Append | Out-Null

        if ($ShutdownOnSuccess) {
            Invoke-Remote "rm -f '$runRoot/shutdown_after_archive.sh' '$runRoot/AUTO_SHUTDOWN' 2>/dev/null || true"
            $target = $User + "@" + $HostName
            ssh -p $Port $target "shutdown -h now"
            "shutdown_sent time=$(Get-Date -Format o)" | Tee-Object -FilePath $watchLog -Append | Out-Null
        }
        if ($RemoveSelfOnSuccess) {
            $self = $PSCommandPath
            Start-Process -WindowStyle Hidden powershell -ArgumentList @(
                "-NoProfile",
                "-Command",
                "Start-Sleep -Seconds 5; Remove-Item -LiteralPath '$self' -Force"
            )
        }
        exit 0
    }
    catch {
        "watch_error=$(Get-Date -Format o) $_" | Tee-Object -FilePath $watchLog -Append | Out-Null
        Start-Sleep -Seconds $PollSeconds
    }
}
