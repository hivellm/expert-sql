"""Test checkpoint-250 of SQL Expert"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# Paths
BASE_MODEL_PATH = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
CHECKPOINT_PATH = "weights/qwen3-06b/checkpoint-250"

print(f"[INFO] Testing SQL Expert - Checkpoint 250")
print(f"[INFO] Base model: {BASE_MODEL_PATH}")
print(f"[INFO] Checkpoint: {CHECKPOINT_PATH}")
print("=" * 70)

# Check if checkpoint exists
if not os.path.exists(CHECKPOINT_PATH):
    print(f"[ERROR] Checkpoint not found at {CHECKPOINT_PATH}")
    exit(1)

print("[1/5] Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
print("[OK] Base model loaded")

print("[2/5] Loading checkpoint adapter...")
model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
print("[OK] Checkpoint adapter loaded")

# Test cases
test_cases = [
    {
        "name": "Simple SELECT",
        "schema": "CREATE TABLE employees (id INTEGER, name VARCHAR, age INTEGER)",
        "question": "Find all employees",
        "expected_keywords": ["SELECT", "FROM", "employees"]
    },
    {
        "name": "COUNT with WHERE",
        "schema": "CREATE TABLE head (age INTEGER)",
        "question": "How many heads of the departments are older than 56?",
        "expected_keywords": ["SELECT", "COUNT", "FROM", "head", "WHERE"]
    },
    {
        "name": "JOIN query",
        "schema": """CREATE TABLE city (City_ID VARCHAR, Population INTEGER);
CREATE TABLE farm_competition (Theme VARCHAR, Host_city_ID VARCHAR)""",
        "question": "Show themes of competitions in cities with population larger than 1000",
        "expected_keywords": ["SELECT", "Theme", "JOIN", "WHERE", "Population"]
    },
    {
        "name": "GROUP BY",
        "schema": "CREATE TABLE employees (department VARCHAR, salary DECIMAL)",
        "question": "What is the average salary by department?",
        "expected_keywords": ["SELECT", "AVG", "GROUP BY", "department"]
    },
    {
        "name": "ORDER BY and LIMIT",
        "schema": "CREATE TABLE products (name VARCHAR, price DECIMAL)",
        "question": "Find top 5 most expensive products",
        "expected_keywords": ["SELECT", "ORDER BY", "LIMIT", "5"]
    }
]

print("\n[3/5] Running test cases...")
print("=" * 70)

passed = 0
failed = 0

for i, test in enumerate(test_cases, 1):
    print(f"\nTest {i}/{len(test_cases)}: {test['name']}")
    print(f"Schema: {test['schema'][:60]}...")
    print(f"Question: {test['question']}")
    
    # Generate SQL using ChatML format (Qwen3 format)
    messages = [
        {"role": "system", "content": f"Dialect: postgres\nSchema:\n{test['schema']}"},
        {"role": "user", "content": test['question']}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    sql = response.strip()
    
    # Remove any trailing text after semicolon
    if ';' in sql:
        sql = sql.split(';')[0] + ';'
    
    print(f"Generated SQL: {sql}")
    
    # Check expected keywords
    sql_upper = sql.upper()
    missing_keywords = []
    for keyword in test['expected_keywords']:
        if keyword.upper() not in sql_upper:
            missing_keywords.append(keyword)
    
    if not missing_keywords and len(sql.strip()) > 5:
        print(f"[PASS] SQL contains expected keywords")
        passed += 1
    else:
        print(f"[FAIL] Missing keywords: {missing_keywords}")
        failed += 1

print("\n" + "=" * 70)
print(f"[4/5] Test Results Summary")
print("=" * 70)
print(f"Total tests: {len(test_cases)}")
print(f"Passed: {passed} ({passed/len(test_cases)*100:.1f}%)")
print(f"Failed: {failed} ({failed/len(test_cases)*100:.1f}%)")

print("\n[5/5] Checkpoint Statistics")
print("=" * 70)
print(f"Checkpoint: checkpoint-250")
print(f"Global step: 250")
print(f"Epoch: 0.25 (25% of 1 epoch)")
print(f"Best eval loss: 0.559")
print("=" * 70)

if passed >= len(test_cases) * 0.8:  # 80% pass rate
    print("\n[SUCCESS] Checkpoint-250 shows good SQL generation capabilities!")
    exit(0)
else:
    print("\n[WARNING] Checkpoint-250 needs more training")
    exit(1)

