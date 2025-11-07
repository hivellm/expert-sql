"""
Qualitative Checkpoint Comparison - Expert SQL

This script runs the same prompts on all available checkpoints
and displays results for qualitative analysis by an external LLM.

Run with: F:/Node/hivellm/expert/cli/venv_windows/Scripts/python.exe compare.py
"""

import sys
import os

# Add experts root directory to path to import template
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

# Import functions from template
from compare_checkpoints_template import (
    detect_device, find_checkpoints, load_base_model, load_checkpoints,
    generate_output, print_separator, print_test_header, print_output, main as template_main
)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

# ============================================================================
# EXPERT-SQL SPECIFIC CONFIGURATION
# ============================================================================

BASE_MODEL_PATH = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
CHECKPOINT_DIR = "weights/qwen3-06b"

GEN_CONFIG = {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "do_sample": True,
}

# ============================================================================
# TEST CASES - EXPERT-SQL
# ============================================================================

test_cases = [
    # Basic SELECT
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
    # COUNT aggregations
    {
        "id": "count_001",
        "category": "aggregation",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE orders (id INT, status VARCHAR, total DECIMAL)",
        "user_prompt": "How many orders were cancelled?",
        "expected_type": "sql"
    },
    {
        "id": "count_002",
        "category": "aggregation",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE head (age INTEGER)",
        "user_prompt": "How many heads are older than 56?",
        "expected_type": "sql"
    },
    # JOINs
    {
        "id": "join_001",
        "category": "join",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE users (id INT, name VARCHAR); CREATE TABLE orders (id INT, user_id INT, amount DECIMAL)",
        "user_prompt": "List users and their orders",
        "expected_type": "sql"
    },
    {
        "id": "join_002",
        "category": "join",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE customers (id INT, name VARCHAR); CREATE TABLE orders (id INT, customer_id INT, total DECIMAL); CREATE TABLE products (id INT, name VARCHAR)",
        "user_prompt": "Show all customers who ordered products",
        "expected_type": "sql"
    },
    # GROUP BY
    {
        "id": "groupby_001",
        "category": "group_by",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE sales (product VARCHAR, quantity INT, revenue DECIMAL)",
        "user_prompt": "Total revenue per product",
        "expected_type": "sql"
    },
    {
        "id": "groupby_002",
        "category": "group_by",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE orders (customer_id INT, total DECIMAL, order_date DATE)",
        "user_prompt": "Average order value by customer",
        "expected_type": "sql"
    },
    # ORDER BY and LIMIT
    {
        "id": "orderby_001",
        "category": "ordering",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE movies (title VARCHAR, rating DECIMAL, year INT)",
        "user_prompt": "Top 5 highest rated movies",
        "expected_type": "sql"
    },
    {
        "id": "orderby_002",
        "category": "ordering",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE employees (name VARCHAR, salary DECIMAL, department VARCHAR)",
        "user_prompt": "Show the 3 highest paid employees",
        "expected_type": "sql"
    },
    # Subqueries
    {
        "id": "subquery_001",
        "category": "subquery",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE employees (name VARCHAR, salary DECIMAL)",
        "user_prompt": "Employees with salary above average",
        "expected_type": "sql"
    },
    {
        "id": "subquery_002",
        "category": "subquery",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE products (id INT, price DECIMAL); CREATE TABLE orders (product_id INT, quantity INT)",
        "user_prompt": "Products that have been ordered",
        "expected_type": "sql"
    },
    # DISTINCT
    {
        "id": "distinct_001",
        "category": "distinct",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE visits (user_id INT, page VARCHAR)",
        "user_prompt": "How many unique users visited?",
        "expected_type": "sql"
    },
    # Date operations
    {
        "id": "date_001",
        "category": "date_operations",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE users (id INT, name VARCHAR, created_at DATE)",
        "user_prompt": "List users registered in 2024",
        "expected_type": "sql"
    },
    {
        "id": "date_002",
        "category": "date_operations",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE orders (id INT, order_date DATE, total DECIMAL)",
        "user_prompt": "Orders from the last 30 days",
        "expected_type": "sql"
    },
    # Window functions
    {
        "id": "window_001",
        "category": "window_functions",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE orders (id INT, customer_id INT, total DECIMAL, order_date DATE)",
        "user_prompt": "Find the 5 most recent orders for each customer",
        "expected_type": "sql"
    },
    # Complex queries
    {
        "id": "complex_001",
        "category": "complex",
        "system_prompt": "Dialect: sql\nSchema:\nCREATE TABLE customers (id INT, name VARCHAR, city VARCHAR); CREATE TABLE orders (id INT, customer_id INT, total DECIMAL, order_date DATE)",
        "user_prompt": "Total sales per city, ordered by total descending",
        "expected_type": "sql"
    }
]

# ============================================================================
# MAIN CODE
# ============================================================================

def main():
    """Main function"""
    device = detect_device()
    
    print_separator()
    print("QUALITATIVE CHECKPOINT COMPARISON - EXPERT SQL")
    print("This script generates outputs for external LLM analysis")
    print("Does not evaluate quality automatically")
    print_separator()
    
    # Find checkpoints
    checkpoints = find_checkpoints(CHECKPOINT_DIR)
    if not checkpoints:
        print(f"ERROR: No checkpoints found in: {CHECKPOINT_DIR}")
        print(f"Checkpoint directory: {os.path.abspath(CHECKPOINT_DIR)}")
        print("\nNote: If training hasn't started yet, checkpoints will appear after training begins.")
        sys.exit(1)
    
    print(f"\nCheckpoints found: {[c[0] for c in checkpoints]}")
    print(f"Total tests: {len(test_cases)}")
    print(f"Device: {device}")
    
    # Load models
    base_model, tokenizer = load_base_model(BASE_MODEL_PATH, device)
    checkpoint_models = load_checkpoints(BASE_MODEL_PATH, checkpoints, device)
    
    # Run tests
    print(f"\n[3/3] Running {len(test_cases)} tests...")
    print_separator()
    
    results = []
    
    for test_idx, test_case in enumerate(test_cases, 1):
        print_test_header(test_case, test_idx, len(test_cases))
        
        # Generate with base model
        base_output = generate_output(
            base_model, tokenizer,
            test_case['system_prompt'],
            test_case['user_prompt'],
            GEN_CONFIG,
            device
        )
        print_output("BASE MODEL", base_output)
        
        # Generate with each checkpoint
        checkpoint_outputs = {}
        for step, model in checkpoint_models.items():
            ckp_output = generate_output(
                model, tokenizer,
                test_case['system_prompt'],
                test_case['user_prompt'],
                GEN_CONFIG,
                device
            )
            checkpoint_outputs[step] = ckp_output
            print_output(f"CHECKPOINT-{step}", ckp_output)
        
        # Store result
        results.append({
            "test_id": test_case.get('id', f'test_{test_idx}'),
            "category": test_case.get('category', 'N/A'),
            "expected_type": test_case.get('expected_type', 'N/A'),
            "system_prompt": test_case['system_prompt'],
            "user_prompt": test_case['user_prompt'],
            "base_output": base_output,
            "checkpoint_outputs": checkpoint_outputs
        })
        
        print_separator()
    
    # Final summary
    print_separator()
    print("\nEXECUTION SUMMARY")
    print_separator()
    print(f"Total tests executed: {len(test_cases)}")
    print(f"Checkpoints tested: {[c[0] for c in checkpoints]}")
    print(f"Base model: {BASE_MODEL_PATH}")
    print(f"\nAll outputs have been displayed above.")
    print("\n" + "="*80)
    print("INSTRUCTIONS FOR LLM ANALYSIS:")
    print("="*80)
    print("Analyze the results above to determine:")
    print("  1. Which checkpoint produces best quality SQL queries")
    print("  2. Which checkpoint should be used to generate the package")
    print("  3. If training is progressing correctly")
    print("  4. Identify common issues (syntax errors, missing clauses, incorrect logic)")
    print("  5. Compare evolution between checkpoints")
    print("  6. Check SQL correctness (valid syntax, proper table/column references)")
    print("="*80)
    
    # Save results to JSON for later analysis
    output_file = "checkpoint_comparison_results.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "expert": "expert-sql",
                "base_model": BASE_MODEL_PATH,
                "checkpoints_tested": [c[0] for c in checkpoints],
                "device": device,
                "test_config": GEN_CONFIG,
                "results": results
            }, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"\nWarning: Could not save results to JSON: {e}")

if __name__ == "__main__":
    main()

