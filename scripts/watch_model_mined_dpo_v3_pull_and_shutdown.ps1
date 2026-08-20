param(
    [Parameter(Mandatory=$true)][string]$HostName,
    [Parameter(Mandatory=$true)][int]$Port,
    [string]$User = "root",
    [string]$RemoteRepo = "/root/autodl-tmp/VLM-Post-Training",
    [int]$PollSeconds = 60,
    [string]$LocalRoot = "outputs\remote_artifacts\model_mined_dpo_v3"
)

$ErrorActionPreference = "Stop"
$Remote = "$User@$HostName"

function Invoke-Remote {
    param([Parameter(Mandatory=$true)][string]$Command)
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        $output = & ssh -p $Port -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no $Remote $Command
        if ($LASTEXITCODE -eq 0) {
            return $output
        }
        if ($attempt -lt 5) {
            Start-Sleep -Seconds 10
        }
    }
    throw "Remote command failed after 5 attempts with exit code $LASTEXITCODE"
}

$runId = (Invoke-Remote -Command "cd '$RemoteRepo' && cat outputs/runtime/model_mined_dpo_v3/LATEST_RUN_ID").Trim()
$runRoot = "outputs/runtime/model_mined_dpo_v3/$runId"
$localDir = Join-Path $LocalRoot $runId
New-Item -ItemType Directory -Force -Path $localDir | Out-Null
Write-Host "Watching model-mined DPO v3 run $runId"

$terminal = $null
while (-not $terminal) {
    try {
        $status = Invoke-Remote -Command "cd '$RemoteRepo' && if test -f '$runRoot/READY_TO_PULL'; then echo READY; elif test -f '$runRoot/FAILED'; then echo FAILED; else echo RUNNING; fi"
        $terminal = switch ($status.Trim()) {
            "READY" { "READY" }
            "FAILED" { "FAILED" }
            default { $null }
        }
        $tail = Invoke-Remote -Command "cd '$RemoteRepo' && tail -n 3 '$runRoot/main.log' 2>/dev/null || true"
        Write-Host "$(Get-Date -Format s) status=$($status.Trim())"
        $tail | ForEach-Object { Write-Host "  $_" }
    } catch {
        Write-Host "$(Get-Date -Format s) status_error=$($_.Exception.Message)"
    }
    if (-not $terminal) {
        Start-Sleep -Seconds $PollSeconds
    }
}

if ($terminal -eq "READY") {
    $archive = (Invoke-Remote -Command "cd '$RemoteRepo' && grep '^archive=' '$runRoot/READY_TO_PULL' | cut -d= -f2-").Trim()
} else {
    $archive = "outputs/archives/model_mined_dpo_v3_$runId.failed.tar.gz"
    Invoke-Remote -Command "cd '$RemoteRepo' && tar -czf '$archive' '$runRoot'"
}

$localArchive = Join-Path $localDir ([System.IO.Path]::GetFileName($archive))
$remoteSpec = "{0}:{1}/{2}" -f $Remote, $RemoteRepo, $archive
Write-Host "Pulling $archive"
& scp -P $Port -o BatchMode=yes -o StrictHostKeyChecking=no $remoteSpec $localArchive
if ($LASTEXITCODE -ne 0) {
    throw "scp failed with exit code $LASTEXITCODE"
}

tar -xzf $localArchive -C $localDir
if ($LASTEXITCODE -ne 0) {
    throw "local archive extraction failed"
}

$decision = Get-ChildItem -Path $localDir -Recurse -Filter final_alignment_decision.json | Select-Object -First 1
$summary = [ordered]@{
    run_id = $runId
    remote_status = $terminal
    pulled_at = (Get-Date).ToString("s")
    archive = (Resolve-Path $localArchive).Path
    extracted_to = (Resolve-Path $localDir).Path
    final_decision = if ($decision) { $decision.FullName } else { $null }
    shutdown_requested = $false
}
$summaryPath = Join-Path $localDir "pull_summary.json"

Write-Host "Requesting server shutdown after verified archive extraction."
try {
    Invoke-Remote -Command "sync; nohup shutdown -h now >/tmp/model_mined_dpo_v3_shutdown.log 2>&1 &"
    $summary.shutdown_requested = $true
} catch {
    # SSH commonly closes while shutdown is succeeding. The archive is already local.
    $summary.shutdown_requested = $true
    $summary.shutdown_response = $_.Exception.Message
}
$summary | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -Path $summaryPath
Write-Host "Pulled artifacts to $localDir and requested shutdown."
