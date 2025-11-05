"""Comparação: Base Model vs Checkpoint-250 vs Checkpoint-500"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re

BASE_MODEL_PATH = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
CHECKPOINT_250 = "weights/qwen3-06b/checkpoint-250"
CHECKPOINT_500 = "weights/qwen3-06b/checkpoint-500"

print("=" * 100)
print("COMPARACAO: BASE vs CHECKPOINT-250 vs CHECKPOINT-500")
print("=" * 100)

print("\n[1/4] Carregando Base Model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

print("[2/4] Carregando Checkpoint-250...")
model_250 = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)
model_250 = PeftModel.from_pretrained(model_250, CHECKPOINT_250)

print("[3/4] Carregando Checkpoint-500...")
model_500 = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)
model_500 = PeftModel.from_pretrained(model_500, CHECKPOINT_500)

# Test cases focados em qualidade
test_cases = [
    {
        "cat": "SELECT",
        "schema": "CREATE TABLE employees (id INT, name VARCHAR, salary DECIMAL)",
        "question": "Liste todos os funcionarios",
        "criteria": ["SELECT", "FROM", "employees"]
    },
    {
        "cat": "WHERE",
        "schema": "CREATE TABLE products (id INT, name VARCHAR, price DECIMAL)",
        "question": "Produtos com preco menor que 50",
        "criteria": ["SELECT", "WHERE", "price", "<", "50"]
    },
    {
        "cat": "COUNT",
        "schema": "CREATE TABLE orders (id INT, status VARCHAR, total DECIMAL)",
        "question": "Quantos pedidos foram cancelados?",
        "criteria": ["SELECT", "COUNT", "WHERE", "status"]
    },
    {
        "cat": "JOIN",
        "schema": "CREATE TABLE users (id INT, name VARCHAR); CREATE TABLE orders (id INT, user_id INT, amount DECIMAL)",
        "question": "Liste usuarios e seus pedidos",
        "criteria": ["SELECT", "JOIN", "users", "orders"]
    },
    {
        "cat": "GROUP BY",
        "schema": "CREATE TABLE sales (product VARCHAR, quantity INT, revenue DECIMAL)",
        "question": "Receita total por produto",
        "criteria": ["SELECT", "SUM", "GROUP BY", "product"]
    },
    {
        "cat": "ORDER BY",
        "schema": "CREATE TABLE movies (title VARCHAR, rating DECIMAL, year INT)",
        "question": "Top 5 filmes melhor avaliados",
        "criteria": ["SELECT", "ORDER BY", "DESC", "LIMIT", "5"]
    },
    {
        "cat": "SUBQUERY",
        "schema": "CREATE TABLE employees (name VARCHAR, salary DECIMAL)",
        "question": "Funcionarios com salario acima da media",
        "criteria": ["SELECT", "WHERE", "salary", ">", "AVG", "SELECT"]
    },
    {
        "cat": "DISTINCT",
        "schema": "CREATE TABLE visits (user_id INT, page VARCHAR)",
        "question": "Quantos usuarios unicos visitaram?",
        "criteria": ["SELECT", "COUNT", "DISTINCT", "user_id"]
    },
    {
        "cat": "LIKE",
        "schema": "CREATE TABLE books (title VARCHAR, author VARCHAR)",
        "question": "Livros com titulo contendo 'Python'",
        "criteria": ["SELECT", "WHERE", "LIKE", "Python"]
    },
    {
        "cat": "IN",
        "schema": "CREATE TABLE students (name VARCHAR, city VARCHAR)",
        "question": "Alunos de Rio ou Sao Paulo",
        "criteria": ["SELECT", "WHERE", "IN", "Rio", "Paulo"]
    },
    {
        "cat": "AGGREGATE",
        "schema": "CREATE TABLE transactions (account INT, amount DECIMAL)",
        "question": "Qual o maior valor de transacao?",
        "criteria": ["SELECT", "MAX", "amount"]
    },
    {
        "cat": "MULTIPLE WHERE",
        "schema": "CREATE TABLE products (name VARCHAR, price DECIMAL, stock INT)",
        "question": "Produtos com preco entre 10 e 100 e estoque maior que 0",
        "criteria": ["SELECT", "WHERE", "BETWEEN", "AND", "stock", ">", "0"]
    }
]

print("[4/4] Gerando SQLs para comparacao...\n")

def evaluate_quality(sql, criteria):
    """Avalia qualidade do SQL baseado em criterios"""
    if not sql or len(sql.strip()) < 10:
        return 0
    
    sql_clean = sql.replace('<think>', '').replace('\n', ' ').strip()
    sql_upper = sql_clean.upper()
    
    # Check if it's explanation instead of SQL
    if any(phrase in sql.lower() for phrase in ['okay', 'let me', 'the user', 'i need to']):
        return 0
    
    # Check basic SQL structure
    if 'SELECT' not in sql_upper or 'FROM' not in sql_upper:
        return 0
    
    # Count criteria matches
    score = 0
    max_score = len(criteria)
    
    for criterion in criteria:
        if criterion.upper() in sql_upper:
            score += 1
    
    # Bonus for clean SQL (no extra text)
    if sql_clean.startswith('SELECT') or sql_clean.startswith('<think>\nSELECT'):
        score += 0.5
    
    # Bonus for ending with semicolon
    if sql_clean.endswith(';'):
        score += 0.5
    
    return (score / max_score) * 10  # Scale to 0-10

results = []

for i, test in enumerate(test_cases, 1):
    messages = [
        {"role": "system", "content": f"Dialect: postgres\nSchema:\n{test['schema']}"},
        {"role": "user", "content": test['question']}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    gen_params = {
        "max_new_tokens": 120,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    # Generate from all models
    with torch.no_grad():
        out_base = base_model.generate(**inputs, **gen_params)
        out_250 = model_250.generate(**inputs, **gen_params)
        out_500 = model_500.generate(**inputs, **gen_params)
    
    sql_base = tokenizer.decode(out_base[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    sql_250 = tokenizer.decode(out_250[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    sql_500 = tokenizer.decode(out_500[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Clean
    if ';' in sql_base: sql_base = sql_base.split(';')[0] + ';'
    if ';' in sql_250: sql_250 = sql_250.split(';')[0] + ';'
    if ';' in sql_500: sql_500 = sql_500.split(';')[0] + ';'
    
    # Evaluate
    score_base = evaluate_quality(sql_base, test['criteria'])
    score_250 = evaluate_quality(sql_250, test['criteria'])
    score_500 = evaluate_quality(sql_500, test['criteria'])
    
    results.append({
        'cat': test['cat'],
        'question': test['question'],
        'score_base': score_base,
        'score_250': score_250,
        'score_500': score_500,
        'sql_base': sql_base[:60],
        'sql_250': sql_250[:60],
        'sql_500': sql_500[:60]
    })

# Print comparison table
print("\n" + "=" * 115)
print("TABELA DE QUALIDADE: BASE vs CHECKPOINT-250 vs CHECKPOINT-500")
print("=" * 115)
print(f"{'#':<3} {'CATEGORIA':<15} {'PERGUNTA':<35} {'BASE':<8} {'CKP-250':<8} {'CKP-500':<8} {'MELHOR':<10}")
print("-" * 115)

total_base = 0
total_250 = 0
total_500 = 0
wins_base = 0
wins_250 = 0
wins_500 = 0

for i, r in enumerate(results, 1):
    total_base += r['score_base']
    total_250 += r['score_250']
    total_500 += r['score_500']
    
    best = max(r['score_base'], r['score_250'], r['score_500'])
    if r['score_base'] == best: 
        best_name = "Base"
        wins_base += 1
    elif r['score_500'] == best:
        best_name = "CKP-500"
        wins_500 += 1
    else:
        best_name = "CKP-250"
        wins_250 += 1
    
    print(f"{i:<3} {r['cat']:<15} {r['question']:<35} {r['score_base']:<8.1f} {r['score_250']:<8.1f} {r['score_500']:<8.1f} {best_name:<10}")

avg_base = total_base / len(results)
avg_250 = total_250 / len(results)
avg_500 = total_500 / len(results)

print("-" * 115)
print(f"{'MEDIA':<3} {'':<15} {'':<35} {avg_base:<8.1f} {avg_250:<8.1f} {avg_500:<8.1f}")
print(f"{'VITORIAS':<3} {'':<15} {'':<35} {wins_base:<8} {wins_250:<8} {wins_500:<8}")
print("=" * 115)

# Summary
print("\n" + "=" * 80)
print("RESUMO COMPARATIVO")
print("=" * 80)
print(f"\nQUALIDADE MEDIA (0-10):")
print(f"  Base Model:      {avg_base:.2f}/10")
print(f"  Checkpoint-250:  {avg_250:.2f}/10  (epoca 0.25)")
print(f"  Checkpoint-500:  {avg_500:.2f}/10  (epoca 0.50)")

print(f"\nMELHORIA vs BASE:")
print(f"  Checkpoint-250:  +{avg_250 - avg_base:.2f} (+{(avg_250 - avg_base)/10*100:.1f}%)")
print(f"  Checkpoint-500:  +{avg_500 - avg_base:.2f} (+{(avg_500 - avg_base)/10*100:.1f}%)")

print(f"\nEVOLUCAO 250 -> 500:")
print(f"  Melhoria:        +{avg_500 - avg_250:.2f} (+{((avg_500 - avg_250)/avg_250)*100 if avg_250 > 0 else 0:.1f}%)")

print(f"\nTAXA DE VITORIA:")
print(f"  Base Model:      {wins_base}/{len(results)} ({wins_base/len(results)*100:.1f}%)")
print(f"  Checkpoint-250:  {wins_250}/{len(results)} ({wins_250/len(results)*100:.1f}%)")
print(f"  Checkpoint-500:  {wins_500}/{len(results)} ({wins_500/len(results)*100:.1f}%)")

# Show examples
print("\n" + "=" * 80)
print("EXEMPLOS COMPARATIVOS")
print("=" * 80)

for idx in [0, 3, 6]:
    r = results[idx]
    print(f"\n[TESTE {idx+1}] {r['cat']}: {r['question']}")
    print(f"\nBase ({r['score_base']:.1f}/10):")
    print(f"  {r['sql_base']}")
    print(f"\nCheckpoint-250 ({r['score_250']:.1f}/10):")
    print(f"  {r['sql_250']}")
    print(f"\nCheckpoint-500 ({r['score_500']:.1f}/10):")
    print(f"  {r['sql_500']}")
    print("-" * 80)

# Final verdict
print("\n" + "=" * 80)
print("VEREDICTO FINAL")
print("=" * 80)

if avg_500 > avg_250 > avg_base:
    print("\n✅ TREINAMENTO PROGREDINDO CORRETAMENTE!")
    print(f"   Checkpoint-500 e {avg_500/avg_250:.2f}x melhor que Checkpoint-250")
    print(f"   Checkpoint-500 e {avg_500/avg_base if avg_base > 0 else 'infinitamente'}x melhor que Base Model")
elif avg_500 > avg_250:
    print("\n✅ CHECKPOINT-500 MELHOR, mas progresso moderado")
    print(f"   Melhoria de {((avg_500-avg_250)/avg_250)*100:.1f}% entre checkpoints")
else:
    print("\n⚠️  CHECKPOINT-500 NAO SUPEROU 250 - possivel overfitting ou plateau")

print(f"\nRECOMENDACAO:")
if avg_500 > 8.0:
    print("  Checkpoint-500 ja esta EXCELENTE (>8.0/10)")
    print("  Considere finalizar treinamento ou fazer fine-tuning fino")
elif avg_500 > 6.0:
    print("  Checkpoint-500 esta BOM (>6.0/10)")
    print("  Continue treinando ate checkpoint-750 ou 1000")
else:
    print("  Checkpoint-500 precisa de mais treinamento")
    print("  Continue ate pelo menos checkpoint-1000")

print("=" * 80)

