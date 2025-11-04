# Test runner for SQL Expert

param(
    [Parameter(Mandatory=$false)]
    [string]$TestSuite = "all"
)

Write-Host "SQL Expert - Test Runner" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan
Write-Host ""

# Ensure we're in the expert directory
$expertDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $expertDir

# Check if Python is available
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python not found" -ForegroundColor Red
    exit 1
}

# Check if tests directory exists
if (-not (Test-Path "tests")) {
    Write-Host "ERROR: tests/ directory not found" -ForegroundColor Red
    exit 1
}

# Determine which tests to run
$testsToRun = @()

switch ($TestSuite.ToLower()) {
    "simple" {
        $testsToRun = @("tests/test_expert.py")
    }
    "comparison" {
        $testsToRun = @("tests/test_comparison.py")
    }
    "hard" {
        $testsToRun = @("tests/test_hard.py")
    }
    "all" {
        $testsToRun = Get-ChildItem -Path "tests" -Filter "test_*.py" | ForEach-Object { $_.FullName }
    }
    default {
        Write-Host "Unknown test suite: $TestSuite" -ForegroundColor Red
        Write-Host "Available: simple, comparison, hard, all" -ForegroundColor Yellow
        exit 1
    }
}

# Run tests
$allPassed = $true
foreach ($testFile in $testsToRun) {
    $testName = Split-Path -Leaf $testFile
    Write-Host "Running $testName..." -ForegroundColor Cyan
    
    python -m pytest $testFile -v
    
    if ($LASTEXITCODE -ne 0) {
        $allPassed = $false
        Write-Host "FAILED: $testName" -ForegroundColor Red
    } else {
        Write-Host "PASSED: $testName" -ForegroundColor Green
    }
    Write-Host ""
}

# Summary
if ($allPassed) {
    Write-Host "All tests passed!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "Some tests failed!" -ForegroundColor Red
    exit 1
}

