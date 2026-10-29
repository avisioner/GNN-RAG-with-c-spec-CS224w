param(
    [string]$EmbeddingModel = 'BAAI/bge-small-en-v1.5',
    [string]$OutputName = 'results_preliminary_bge_small',
    [ValidateSet('query_code', 'question')]
    [string]$QueryMode = 'query_code',
    [double]$DenseWeight = 1.0,
    [switch]$SkipRerank,
    [switch]$Offline
)
$ErrorActionPreference = 'Stop'
$baselineDir = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $baselineDir '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create baseline/.venv and install requirements-preliminary-lock.txt first (CPU PyTorch index required).'
}
$env:HF_HOME = Join-Path $baselineDir '.cache/huggingface'
$env:HF_HUB_DISABLE_XET = '1'
$env:PYTHONUTF8 = '1'
if ($Offline) { $env:HF_HUB_OFFLINE = '1' }
try {
    $rerankArgs = @()
    if (-not $SkipRerank) { $rerankArgs = @('--rerank') }
    & $pythonPath -u (Join-Path $PSScriptRoot 'retrieval_baselines.py') `
        --embedding-model $EmbeddingModel --lexical-mode content @rerankArgs `
        --query-mode $QueryMode --dense-weight $DenseWeight `
        --top-k 5 10 50 --candidate-k 50 --threads 4 `
        --output-dir (Join-Path $PSScriptRoot $OutputName)
    if ($LASTEXITCODE -ne 0) { throw "Baseline exited with code $LASTEXITCODE" }
} finally {
    if ($Offline) { Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue }
}
