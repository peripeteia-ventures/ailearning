$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$sourcePath = Join-Path $projectRoot 'examples/ai-farm'
$destinationFolder = Join-Path $projectRoot 'public/examples'
New-Item -ItemType Directory -Force -Path $destinationFolder | Out-Null
$files = Get-ChildItem -LiteralPath $sourcePath -Recurse -File -Force
if ($files | Where-Object { $_.Name -eq '.env' -or $_.FullName -match '[\\/]node_modules[\\/]' }) { throw 'Refusing to bundle local environment secrets or node_modules.' }
Compress-Archive -LiteralPath $sourcePath -DestinationPath (Join-Path $destinationFolder 'ai-farm.zip') -Force
Write-Host 'Created public/examples/ai-farm.zip. Rebuild to include it in the running app.'
