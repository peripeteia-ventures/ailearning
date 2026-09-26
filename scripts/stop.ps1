$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$recordPath = Join-Path $projectRoot 'work/server-process.json'
if (-not (Test-Path -LiteralPath $recordPath)) { Write-Host 'No launcher-managed server is recorded.'; exit 0 }
$record = Get-Content -LiteralPath $recordPath | ConvertFrom-Json
$process = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
if (-not $process) { Write-Host 'Recorded server has already stopped.'; exit 0 }
$expectedEntry = [System.IO.Path]::GetFullPath((Join-Path $projectRoot 'server/index.ts'))
if ($record.entry -ne $expectedEntry -or $process.Path -ne $record.executable -or $process.StartTime.ToUniversalTime().ToString('o') -ne $record.started) { throw 'Recorded process identity no longer matches. Nothing was stopped.' }
Stop-Process -Id $process.Id
Remove-Item -LiteralPath $recordPath
Write-Host 'Latent stopped. PostgreSQL and Tailscale routes were left running.'
