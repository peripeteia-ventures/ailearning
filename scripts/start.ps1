param([switch]$NoBrowser, [switch]$NoBuild)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$url = 'http://127.0.0.1:3002'
try {
  $health = Invoke-RestMethod "$url/api/health" -TimeoutSec 3
  if ($health.app -eq 'latent' -and $health.ok) {
    Write-Host "Latent is already running at $url"
    if (-not $NoBrowser) { Start-Process $url }
    exit 0
  }
} catch {}
& npm.cmd run db:setup
if ($LASTEXITCODE -ne 0) { throw 'Database setup failed.' }
if (-not $NoBuild -or -not (Test-Path -LiteralPath (Join-Path $projectRoot 'dist/index.html'))) {
  & npm.cmd run build
  if ($LASTEXITCODE -ne 0) { throw 'Build failed.' }
}
$workPath = Join-Path $projectRoot 'work'
New-Item -ItemType Directory -Force -Path $workPath | Out-Null
$nodePath = (Get-Command node.exe).Source
$process = Start-Process -FilePath $nodePath -ArgumentList 'server/index.ts' -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $workPath 'server.log') -RedirectStandardError (Join-Path $workPath 'server-error.log') -PassThru
@{ pid = $process.Id; started = $process.StartTime.ToUniversalTime().ToString('o'); executable = $nodePath; entry = (Join-Path $projectRoot 'server/index.ts') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $workPath 'server-process.json')
for ($attempt = 0; $attempt -lt 30; $attempt++) {
  Start-Sleep -Milliseconds 400
  try {
    $health = Invoke-RestMethod "$url/api/health" -TimeoutSec 2
    if ($health.app -eq 'latent' -and $health.ok) {
      Write-Host "Latent is ready at $url"
      if (-not $NoBrowser) { Start-Process $url }
      exit 0
    }
  } catch {}
  if ($process.HasExited) { throw 'Server exited. Check work/server-error.log.' }
}
throw 'Server did not become ready. Check work/server-error.log.'
