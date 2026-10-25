$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'SCOPE.md'))) {
    throw 'Expected the learning-certify project root'
}
Push-Location -LiteralPath $projectRoot
try {
    Get-Command uv -ErrorAction Stop | Out-Null
    uv python install 3.13 --install-dir '.tools\python' --no-bin --no-registry --cache-dir '.cache\uv'
    if ($LASTEXITCODE -ne 0) { throw 'Python installation failed' }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.tools\python'
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        uv venv --python 3.13 --managed-python --cache-dir '.cache\uv' .venv
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
    }
    uv pip install --python '.venv\Scripts\python.exe' --cache-dir '.cache\uv' -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Pinned dependency installation failed' }
    & '.venv\Scripts\python.exe' 'scripts\run_experiment.py' doctor
    if ($LASTEXITCODE -ne 0) { throw 'Environment verification failed' }
} finally {
    Pop-Location
}
