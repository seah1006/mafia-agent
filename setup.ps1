# Run from PowerShell after installing Python 3.11.9 and Ollama.
[CmdletBinding()]
param([string]$PythonExe, [switch]$SkipModels)

$ErrorActionPreference = 'Stop'
$want = '3.11.9'
$venv = Join-Path $PSScriptRoot '.venv'
$localPython = Join-Path $venv 'Scripts\python.exe'
$versionCode = 'import platform; print(platform.python_version())'

function Assert-Version([string]$Executable) {
    $actual = & $Executable -I -c $versionCode
    if ($LASTEXITCODE -ne 0 -or $actual -ne $want) {
        throw "Expected Python $want at '$Executable'; got '$actual'."
    }
}

if (-not (Test-Path -LiteralPath $localPython)) {
    if (-not $PythonExe) {
        $knownPath = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'
        if (Test-Path -LiteralPath $knownPath) {
            $PythonExe = $knownPath
        } elseif (Get-Command py -ErrorAction SilentlyContinue) {
            $PythonExe = & py -3.11 -I -c 'import sys; print(sys.executable)'
            if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.11.9 or pass -PythonExe.' }
        } else {
            $PythonExe = 'python'
        }
    }
    Assert-Version $PythonExe
    & $PythonExe -I -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Creating .venv failed.' }
}
Assert-Version $localPython

# Isolated Python and pip flags keep global Python/PIP_PREFIX settings out.
& $localPython -I -m pip --isolated install --requirement (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Installing Python dependencies failed.' }

if (-not $SkipModels) {
    $ollama = Get-Command ollama -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $ollama) { throw 'Install Ollama for Windows and reopen PowerShell.' }
    try {
        $tags = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 5
    } catch {
        throw 'Start the Ollama app, then run setup.ps1 again.'
    }
    foreach ($model in @('qwen3:8b', 'bge-m3')) {
        if ($model -notin $tags.models.name) {
            & $ollama.Source pull $model
            if ($LASTEXITCODE -ne 0) { throw "Failed to download $model." }
        }
    }
}

Write-Host "Setup complete. Select this interpreter in VS Code: $localPython"
Write-Host 'Open 01_create_db.ipynb first to build the local vector database.'
