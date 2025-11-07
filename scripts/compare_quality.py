"""Compare SQL quality: Base Model vs Expert (Checkpoint-250)"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_PATH = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
CHECKPOINT_PATH = "weights/qwen3-06b/checkpoint-250"

print("=" * 80)
print("ANALISE QUALITATIVA: BASE MODEL vs EXPERT-SQL (Checkpoint-250)")
print("=" * 80)

# Load base model
print("\n[1/3] Carregando base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

# Load expert model
print("[2/3] Carregando expert model (checkpoint-250)...")
expert_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
expert_model = PeftModel.from_pretrained(expert_model, CHECKPOINT_PATH)

# Test cases
test_cases = [
    {
        "name": "SELECT Simples",
        "schema": "CREATE TABLE employees (id INTEGER, name VARCHAR, age INTEGER, salary DECIMAL)",
        "question": "Mostre todos os funcionarios"
    },
    {
        "name": "Agregacao com WHERE",
        "schema": "CREATE TABLE head (name VARCHAR, age INTEGER, born_state VARCHAR)",
        "question": "Quantos chefes de departamento tem mais de 56 anos?"
    },
    {
        "name": "JOIN entre tabelas",
        "schema": """CREATE TABLE city (City_ID VARCHAR, Official_Name VARCHAR, Population INTEGER);
CREATE TABLE farm_competition (Competition_ID VARCHAR, Theme VARCHAR, Host_city_ID VARCHAR)""",
        "question": "Mostre os temas das competicoes em cidades com populacao maior que 1000"
    },
    {
        "name": "GROUP BY com agregacao",
        "schema": "CREATE TABLE employees (department VARCHAR, name VARCHAR, salary DECIMAL)",
        "question": "Qual o salario medio por departamento?"
    },
    {
        "name": "Subconsulta",
        "schema": """CREATE TABLE products (product_id INTEGER, name VARCHAR, price DECIMAL, category VARCHAR);
CREATE TABLE orders (order_id INTEGER, product_id INTEGER, quantity INTEGER)""",
        "question": "Quais produtos nunca foram pedidos?"
    }
]

print("[3/3] Gerando SQLs para comparacao...\n")

for i, test in enumerate(test_cases, 1):
    print("\n" + "=" * 80)
    print(f"TESTE {i}: {test['name']}")
    print("=" * 80)
    print(f"\nSchema:")
    print(test['schema'])
    print(f"\nPergunta: {test['question']}")
    
    messages = [
        {"role": "system", "content": f"Dialect: postgres\nSchema:\n{test['schema']}"},
        {"role": "user", "content": test['question']}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    # BASE MODEL
    print("\n" + "-" * 40)
    print("BASE MODEL (sem expert):")
    print("-" * 40)
    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    base_sql = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    if ';' in base_sql:
        base_sql = base_sql.split(';')[0] + ';'
    print(base_sql.strip())
    
    # EXPERT MODEL
    print("\n" + "-" * 40)
    print("EXPERT MODEL (checkpoint-250):")
    print("-" * 40)
    with torch.no_grad():
        outputs = expert_model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    expert_sql = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    if ';' in expert_sql:
        expert_sql = expert_sql.split(';')[0] + ';'
    print(expert_sql.strip())
    
    # ANALISE
    print("\n" + "-" * 40)
    print("ANALISE QUALITATIVA:")
    print("-" * 40)
    
    base_upper = base_sql.upper()
    expert_upper = expert_sql.upper()
    
    # Criterios de qualidade
    criterios = {
        "SELECT presente": "SELECT" in expert_upper,
        "FROM presente": "FROM" in expert_upper,
        "Sintaxe SQL valida": "SELECT" in expert_upper and "FROM" in expert_upper,
        "Usa schema correto": any(word in expert_sql.lower() for word in test['schema'].lower().split()),
    }
    
    print("Expert:")
    for criterio, passou in criterios.items():
        status = "SIM" if passou else "NAO"
        print(f"  - {criterio}: {status}")
    
    # Comparacao direta
    if len(expert_sql.strip()) > len(base_sql.strip()) * 0.5:
        print("\nConclusao: Expert gerou SQL mais completo")
    else:
        print("\nConclusao: Base model gerou resposta mais concisa")

print("\n" + "=" * 80)
print("FIM DA ANALISE")
print("=" * 80)

