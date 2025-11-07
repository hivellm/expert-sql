# PowerShell script to download The Stack SQL dataset
# Requires HuggingFace token (set via environment variable or huggingface-cli login)

$ErrorActionPreference = "Stop"

# Check if token is set
if (-not $env:HF_TOKEN -and -not $env:HUGGINGFACE_TOKEN) {
    Write-Host "[ERROR] HuggingFace token not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please set the token using one of these methods:"
    Write-Host "  1. Set environment variable: `$env:HF_TOKEN = 'your_token_here'"
    Write-Host "  2. Use huggingface-cli login"
    Write-Host ""
    Write-Host "Get your token from: https://huggingface.co/settings/tokens"
    Write-Host "Accept dataset terms at: https://huggingface.co/datasets/bigcode/the-stack"
    exit 1
}

# Navigate to expert-sql directory
Set-Location "F:\Node\hivellm\expert\experts\expert-sql"

Write-Host "========================================"
Write-Host "Downloading The Stack SQL Dataset"
Write-Host "========================================"
Write-Host ""
Write-Host "Parameters:"
Write-Host "  Limit: 50,000 files with SQL"
Write-Host "  Max Check: 1,000,000 files"
Write-Host "  Output: datasets/the_stack_sql.jsonl"
Write-Host ""

# Run the download script
& "F:\Node\hivellm\expert\cli\venv_windows\Scripts\python.exe" `
    scripts/download_the_stack_sql.py `
    --limit 10000 `
    --max-check 200000 `
    --output datasets/the_stack_sql.jsonl

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Download failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "[OK] Download completed successfully!"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  To merge with current dataset:"
Write-Host "    F:\Node\hivellm\expert\cli\venv_windows\Scripts\python.exe scripts/merge_the_stack_sql.py --backup"

