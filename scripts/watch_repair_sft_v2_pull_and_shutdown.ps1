param(
    [Parameter(Mandatory=$true)][string]$HostName,
    [Parameter(Mandatory=$true)][int]$Port,
    [string]$User = "root",
    [string]$RemoteRepo = "/root/autodl-tmp/VLM-Post-Training",
    [int]$MaxWaitMinutes = 180,
    [int]$PollSeconds = 120,
    [string]$LocalArchiveRoot = "outputs\remote_artifacts\phase08_high_risk_repair_r2_order_id_train_decode_dev",
    [switch]$ShutdownOnTimeout
)

$ErrorActionPreference = "Stop"

$Remote = "$User@$HostName"

$RunDir = "outputs/runtime/high_risk_repair_sft_v2/repair_sft_v2_order_id_20260815_212900"
$PidFile = "$RunDir/decode_eval.pid"
$DecodeLog = "$RunDir/logs/decode_eval.log"
$TrainLog = "$RunDir/logs/train.log"

$Predictions = "outputs/predictions/phase08_high_risk_repair_r2_order_id_train_decode_dev/repair_sft_r2/train_decode_dev.jsonl"
$GroundTruth = "outputs/eval_sets/phase08_high_risk_repair_r2_order_id_train_decode_dev/train_decode_dev.jsonl"
$Metrics = "outputs/eval_reports/phase08_high_risk_repair_r2_order_id_train_decode_dev/repair_sft_r2_train_decode_dev_metrics.json"
$Errors = "outputs/eval_reports/phase08_high_risk_repair_r2_order_id_train_decode_dev/repair_sft_r2_train_decode_dev_errors.jsonl"
$Config = "configs/train/high_risk_repair_sft_r2_order_id_from_r1_existing_images_qwen3vl_8b_server.yaml"
$Manifest = "docs/experiments/phase08_high_risk_repair_pack_20260813/repair_sft_v2_order_id_mix_manifest.json"

$LocalArchive = Join-Path $LocalArchiveRoot (Get-Date -Format "yyyyMMdd_HHmmss")
New-Item -ItemType Directory -Force -Path $LocalArchive | Out-Null

function Invoke-Remote {
    param([Parameter(Mandatory=$true)][string]$Command)

    & ssh -p $Port -o BatchMode=yes -o StrictHostKeyChecking=no $Remote $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed with exit code $LASTEXITCODE"
    }
}

function Test-RemoteFile {
    param([Parameter(Mandatory=$true)][string]$Path)

    & ssh -p $Port -o BatchMode=yes -o StrictHostKeyChecking=no $Remote "cd '$RemoteRepo' && test -f '$Path'"
    return ($LASTEXITCODE -eq 0)
}

function Copy-RemoteFileIfPresent {
    param(
        [Parameter(Mandatory=$true)][string]$RemotePath,
        [Parameter(Mandatory=$true)][string]$DestinationRoot
    )

    if (Test-RemoteFile -Path $RemotePath) {
        $relativeLocalPath = $RemotePath -replace '/', [System.IO.Path]::DirectorySeparatorChar
        $destinationPath = Join-Path $DestinationRoot $relativeLocalPath
        $destinationDir = Split-Path -Parent $destinationPath
        New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null
        Write-Host "Pulling $RemotePath"
        & scp -P $Port -o BatchMode=yes -o StrictHostKeyChecking=no "${Remote}:$RemoteRepo/$RemotePath" $destinationPath
        if ($LASTEXITCODE -ne 0) {
            throw "scp failed for $RemotePath with exit code $LASTEXITCODE"
        }
    } else {
        Write-Host "Missing on remote, skipped: $RemotePath"
    }
}

function Get-RemoteStatus {
    $statusScript = @"
cd '$RemoteRepo'
pid=`$(cat '$PidFile' 2>/dev/null || true)
running=0
if [ -n "`$pid" ] && ps -p "`$pid" >/dev/null 2>&1; then running=1; fi
pred_count=0
if [ -f '$Predictions' ]; then pred_count=`$(wc -l < '$Predictions'); fi
metrics=0
if [ -s '$Metrics' ]; then metrics=1; fi
errors=0
if [ -s '$Errors' ]; then errors=1; fi
echo "pid=`$pid"
echo "running=`$running"
echo "pred_count=`$pred_count"
echo "metrics=`$metrics"
echo "errors=`$errors"
"@

    $lines = Invoke-Remote -Command $statusScript
    $result = @{}
    foreach ($line in $lines) {
        if ($line -match "^(?<key>[^=]+)=(?<value>.*)$") {
            $result[$Matches.key] = $Matches.value
        }
    }
    return $result
}

$deadline = (Get-Date).AddMinutes($MaxWaitMinutes)
$completed = $false
$timedOut = $false
$sshFailures = 0
$maxSshFailures = 20

Write-Host "Watching remote decode/eval for up to $MaxWaitMinutes minutes."
while ($true) {
    try {
        $status = Get-RemoteStatus
        $sshFailures = 0
    } catch {
        $sshFailures += 1
        Write-Host "$(Get-Date -Format s) remote_status_error attempt=$sshFailures/$maxSshFailures message=$($_.Exception.Message)"
        if ($sshFailures -ge $maxSshFailures) {
            throw
        }
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    $line = "remote_status pid={0} running={1} pred_count={2}/152 metrics={3} errors={4}" -f `
        $status.pid, $status.running, $status.pred_count, $status.metrics, $status.errors
    Write-Host "$(Get-Date -Format s) $line"

    if (($status.running -eq "0") -and ($status.metrics -eq "1") -and ($status.errors -eq "1")) {
        $completed = $true
        break
    }

    if ((Get-Date) -ge $deadline) {
        $timedOut = $true
        break
    }

    Start-Sleep -Seconds $PollSeconds
}

$filesToPull = @(
    $Predictions,
    $GroundTruth,
    $Metrics,
    $Errors,
    $DecodeLog,
    $TrainLog,
    $PidFile,
    $Config,
    $Manifest
)

foreach ($path in $filesToPull) {
    Copy-RemoteFileIfPresent -RemotePath $path -DestinationRoot $LocalArchive
}

$summaryPath = Join-Path $LocalArchive "watch_summary.json"
$summary = [ordered]@{
    pulled_at = (Get-Date).ToString("s")
    remote = $Remote
    port = $Port
    remote_repo = $RemoteRepo
    local_archive = (Resolve-Path $LocalArchive).Path
    completed = $completed
    timed_out = $timedOut
    shutdown_requested = $false
    shutdown_reason = $null
}

if ($completed -or ($timedOut -and $ShutdownOnTimeout)) {
    $summary.shutdown_requested = $true
    $summary.shutdown_reason = if ($completed) { "decode_eval_completed" } else { "timeout_with_ShutdownOnTimeout" }
    Write-Host "Requesting remote shutdown: $($summary.shutdown_reason)"
    Invoke-Remote -Command "sync; nohup shutdown -h now >/tmp/codex_shutdown.log 2>&1 &"
} else {
    $summary.shutdown_reason = "not_complete_timeout_guard"
    Write-Host "Remote shutdown skipped because the job did not complete. Pass -ShutdownOnTimeout to override this guard."
}

$summary | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -Path $summaryPath
Write-Host "Archive written to $LocalArchive"
