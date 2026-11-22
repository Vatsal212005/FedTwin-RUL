param(
    [switch]$RegenerateData
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$frontendRoot = Join-Path $repoRoot "app\frontend"
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$npmExe = (Get-Command npm.cmd -ErrorAction Stop).Source

if ($RegenerateData) {
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        throw "Repository virtual environment not found at $pythonExe"
    }

    & $pythonExe (Join-Path $PSScriptRoot "generate_dashboard_data.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Dashboard data generation failed with exit code $LASTEXITCODE"
    }
}

Push-Location $frontendRoot
try {
    if (-not (Test-Path -LiteralPath (Join-Path $frontendRoot "node_modules"))) {
        & $npmExe install
        if ($LASTEXITCODE -ne 0) {
            throw "npm install failed with exit code $LASTEXITCODE"
        }
    }

    & $npmExe run dev -- --host 127.0.0.1
}
finally {
    Pop-Location
}
