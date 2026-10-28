param(
    [string]$DataDir = $env:TEDDY_DATA_DIR,
    [switch]$SkipInstall,
    [switch]$SkipHistorical
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runtime = Join-Path $Root 'runtime'
$Python = Join-Path $Runtime '.venv\Scripts\python.exe'
$Requirements = Join-Path $Root 'requirements.txt'
$Runner = Join-Path $Root 'run_pipeline.py'

if ($DataDir) {
    $ResolvedDataDir = (Resolve-Path -LiteralPath $DataDir).Path
    $env:TEDDY_DATA_DIR = $ResolvedDataDir
}

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

Push-Location $Runtime
try {
    if (-not (Test-Path -LiteralPath $Python)) {
        Write-Host 'Creating runtime virtual environment...'
        python -m venv .venv
        $SkipInstall = $false
    }

    if (-not $SkipInstall) {
        Write-Host 'Installing Python dependencies...'
        & $Python -m pip install -r $Requirements
        if ($LASTEXITCODE -ne 0) {
            throw 'Dependency installation failed.'
        }
    }

    $RunnerArgs = @($Runner)
    if ($SkipHistorical) {
        $RunnerArgs += '--skip-historical'
    }
    & $Python @RunnerArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
