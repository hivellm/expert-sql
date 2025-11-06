#!/usr/bin/env pwsh
# Test script for packaged expert-sql inference
# Tests loading from .expert package and running inference

$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Testing Packaged Expert-SQL Inference" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$PackageFile = "expert-sql-qwen3-0-6b.v0.2.0.expert"
$TestDir = "test_package_inference"
$BaseModelPath = "F:/Node/hivellm/expert/models/Qwen3-0.6B"

# Cleanup old test directory
if (Test-Path $TestDir) {
    Write-Host "Cleaning up old test directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $TestDir
}

# Check if package exists
if (-not (Test-Path $PackageFile)) {
    Write-Host "ERROR: Package not found: $PackageFile" -ForegroundColor Red
    Write-Host "Run packaging first: ..\..\cli\target\release\expert-cli package --manifest manifest.json --weights weights" -ForegroundColor Yellow
    exit 1
}

# Extract package
Write-Host "[1/4] Extracting package..." -ForegroundColor Green
New-Item -ItemType Directory -Path $TestDir | Out-Null
tar -xzf $PackageFile -C $TestDir

# List extracted files
Write-Host ""
Write-Host "Extracted files:" -ForegroundColor Cyan
Get-ChildItem $TestDir | Select-Object Name, Length | Format-Table

# Validate structure
Write-Host "[2/4] Validating package structure..." -ForegroundColor Green

$RequiredFiles = @(
    "manifest.json",
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json"
)

$AllValid = $true
foreach ($file in $RequiredFiles) {
    $filePath = Join-Path $TestDir $file
    if (Test-Path $filePath) {
        Write-Host "  [OK] $file" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] $file - NOT FOUND" -ForegroundColor Red
        $AllValid = $false
    }
}

if (-not $AllValid) {
    Write-Host ""
    Write-Host "ERROR: Package structure validation failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[3/4] Testing inference with Python..." -ForegroundColor Green

# Create test script
$PythonTest = @"
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "$BaseModelPath",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("$BaseModelPath", trust_remote_code=True)

print("Loading adapter from extracted package...")
model = PeftModel.from_pretrained(base_model, "$TestDir")

print("")
print("="*80)
print("TESTING SQL GENERATION")
print("="*80)

# Test cases
test_cases = [
    {
        "schema": "CREATE TABLE users (id INT, name VARCHAR, email VARCHAR, created_at DATE)",
        "question": "Liste todos os usuários cadastrados nos últimos 30 dias"
    },
    {
        "schema": "CREATE TABLE products (id INT, name VARCHAR, price DECIMAL, stock INT)",
        "question": "Mostre produtos com estoque maior que zero ordenados por preço"
    },
    {
        "schema": "CREATE TABLE orders (id INT, customer_id INT, total DECIMAL, status VARCHAR); CREATE TABLE customers (id INT, name VARCHAR, email VARCHAR)",
        "question": "Liste os 10 clientes que mais gastaram com nome e total"
    }
]

for i, test in enumerate(test_cases, 1):
    print(f"\n[TEST {i}/3]")
    print(f"Schema: {test['schema'][:60]}...")
    print(f"Question: {test['question']}")
    print("-" * 80)
    
    messages = [
        {"role": "system", "content": f"Dialect: postgres\nSchema:\n{test['schema']}"},
        {"role": "user", "content": test['question']}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    sql = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Clean output
    if ';' in sql:
        sql = sql.split(';')[0] + ';'
    sql = sql.replace('<think>\n', '').replace('<think>', '').strip()
    
    # Validate
    is_valid = sql.upper().startswith('SELECT') or 'SELECT' in sql.upper()[:50]
    status = "[OK]" if is_valid else "[FAIL]"
    
    print(f"{status} {sql}")

print("")
print("="*80)
print("INFERENCE TEST COMPLETE!")
print("="*80)
"@

$PythonTest | Out-File -FilePath "test_inference.py" -Encoding UTF8

# Run test with venv
Write-Host ""
F:\Node\hivellm\expert\cli\venv_windows\Scripts\python.exe test_inference.py

Write-Host ""
Write-Host "[4/4] Cleanup..." -ForegroundColor Green
Remove-Item -Force test_inference.py
Remove-Item -Recurse -Force $TestDir

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Test Complete!" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

