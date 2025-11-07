"""
Compare checkpoint-250 with previous version (v0.2.1 package)
"""
import sys
import os
import tarfile
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from compare_checkpoints_template import (
    detect_device, generate_output, print_separator, print_test_header, print_output
)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_PATH = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
CHECKPOINT_DIR = "weights/qwen3-06b"
PACKAGE_PATH = "expert-sql-qwen3-0-6b.v0.2.1.expert"

GEN_CONFIG = {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "do_sample": True,
}

# Test cases from compare.py
test_cases = [
    {
        "id": "select_001",
        "category": "basic_select",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE employees (id INT, name VARCHAR, salary DECIMAL)",
        "user_prompt": "List all employees",
        "expected_type": "sql"
    },
    {
        "id": "select_002",
        "category": "where_clause",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE products (id INT, name VARCHAR, price DECIMAL)",
        "user_prompt": "Find products with price less than 50",
        "expected_type": "sql"
    },
    {
        "id": "count_001",
        "category": "aggregation",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE orders (id INT, status VARCHAR, total DECIMAL)",
        "user_prompt": "How many orders were cancelled?",
        "expected_type": "sql"
    },
    {
        "id": "join_001",
        "category": "join",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE users (id INT, name VARCHAR); CREATE TABLE orders (id INT, user_id INT, amount DECIMAL)",
        "user_prompt": "List users and their orders",
        "expected_type": "sql"
    },
    {
        "id": "groupby_001",
        "category": "group_by",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE sales (product VARCHAR, quantity INT, revenue DECIMAL)",
        "user_prompt": "Total revenue per product",
        "expected_type": "sql"
    },
    {
        "id": "subquery_001",
        "category": "subquery",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE employees (name VARCHAR, salary DECIMAL)",
        "user_prompt": "Employees with salary above average",
        "expected_type": "sql"
    },
    {
        "id": "complex_001",
        "category": "complex",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE customers (id INT, name VARCHAR, city VARCHAR); CREATE TABLE orders (id INT, customer_id INT, total DECIMAL, order_date DATE)",
        "user_prompt": "Total sales per city, ordered by total descending",
        "expected_type": "sql"
    }
]

def extract_package(package_path):
    """Extract package to temporary directory"""
    temp_dir = tempfile.mkdtemp()
    print(f"Extracting package to: {temp_dir}")
    
    with tarfile.open(package_path, 'r:gz') as tar:
        tar.extractall(temp_dir)
    
    return temp_dir

def load_package_adapter(base_model, package_dir, device):
    """Load adapter from extracted package"""
    adapter_path = os.path.join(package_dir, "adapter_model.safetensors")
    adapter_config = os.path.join(package_dir, "adapter_config.json")
    
    if not os.path.exists(adapter_path) or not os.path.exists(adapter_config):
        # Try alternative structure
        adapter_path = os.path.join(package_dir, "weights", "adapter", "adapter_model.safetensors")
        adapter_config = os.path.join(package_dir, "weights", "adapter", "adapter_config.json")
    
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter not found in package: {adapter_path}")
    
    # Create a temporary checkpoint-like structure
    temp_checkpoint = os.path.join(package_dir, "checkpoint-temp")
    os.makedirs(temp_checkpoint, exist_ok=True)
    
    # Copy adapter files
    shutil.copy(adapter_path, os.path.join(temp_checkpoint, "adapter_model.safetensors"))
    shutil.copy(adapter_config, os.path.join(temp_checkpoint, "adapter_config.json"))
    
    # Load adapter
    model = PeftModel.from_pretrained(base_model, temp_checkpoint)
    return model

def main():
    device = detect_device()
    
    print_separator()
    print("COMPARISON: Checkpoint-250 vs Previous Version (v0.2.1)")
    print_separator()
    
    # Load base model
    print("\n[1/4] Loading Base Model...")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    device_map = "auto" if device == "cuda" else None
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=True
    )
    if device == "cpu":
        base_model = base_model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    print("[OK] Base Model loaded")
    
    # Load checkpoint-250
    print("\n[2/4] Loading checkpoint-250...")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint-250")
    model_250 = PeftModel.from_pretrained(base_model, checkpoint_path)
    print("[OK] Checkpoint-250 loaded")
    
    # Load previous version from package
    print("\n[3/4] Loading previous version (v0.2.1) from package...")
    if not os.path.exists(PACKAGE_PATH):
        print(f"ERROR: Package not found: {PACKAGE_PATH}")
        sys.exit(1)
    
    package_dir = extract_package(PACKAGE_PATH)
    try:
        model_previous = load_package_adapter(base_model, package_dir, device)
        print("[OK] Previous version loaded")
    except Exception as e:
        print(f"ERROR loading previous version: {e}")
        shutil.rmtree(package_dir, ignore_errors=True)
        sys.exit(1)
    
    # Run tests
    print(f"\n[4/4] Running {len(test_cases)} comparison tests...")
    print_separator()
    
    results = []
    
    for test_idx, test_case in enumerate(test_cases, 1):
        print_test_header(test_case, test_idx, len(test_cases))
        
        # Generate with checkpoint-250
        output_250 = generate_output(
            model_250, tokenizer,
            test_case['system_prompt'],
            test_case['user_prompt'],
            GEN_CONFIG,
            device
        )
        print_output("CHECKPOINT-250 (NEW)", output_250)
        
        # Generate with previous version
        output_previous = generate_output(
            model_previous, tokenizer,
            test_case['system_prompt'],
            test_case['user_prompt'],
            GEN_CONFIG,
            device
        )
        print_output("V0.2.1 (PREVIOUS)", output_previous)
        
        results.append({
            "test_id": test_case.get('id', f'test_{test_idx}'),
            "category": test_case.get('category', 'N/A'),
            "checkpoint_250": output_250,
            "v0_2_1": output_previous
        })
        
        print_separator()
    
    # Cleanup
    shutil.rmtree(package_dir, ignore_errors=True)
    
    # Summary
    print_separator()
    print("\nCOMPARISON SUMMARY")
    print_separator()
    print(f"Total tests: {len(test_cases)}")
    print(f"Checkpoint-250: New training checkpoint")
    print(f"V0.2.1: Previous production version")
    print("\nAnalyze the results above to determine:")
    print("  1. If checkpoint-250 shows improvement over v0.2.1")
    print("  2. If checkpoint-250 maintains quality from previous version")
    print("  3. If training is progressing correctly")
    print("  4. Identify any regressions or improvements")
    print_separator()

if __name__ == "__main__":
    main()

