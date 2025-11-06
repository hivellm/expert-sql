"""
Teste de Qualidade - Queries SQL do Mundo Real
Checkpoint: 1250
Foco: Queries práticas que programadores usam no dia a dia
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from datetime import datetime

BASE_MODEL_PATH = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
CHECKPOINT_1250 = "weights/qwen3-06b/checkpoint-1250"

print("=" * 100)
print("TESTE DE QUALIDADE - QUERIES SQL DO MUNDO REAL")
print(f"Checkpoint: 1250")
print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)

print("\nCarregando modelo base...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

print("Carregando checkpoint 1250...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)
model = PeftModel.from_pretrained(model, CHECKPOINT_1250)

# Cenários práticos do mundo real
real_world_scenarios = [
    # === E-COMMERCE ===
    {
        "category": "E-commerce",
        "name": "Listar produtos em estoque",
        "difficulty": "Básico",
        "schema": "CREATE TABLE products (id INT, name VARCHAR, price DECIMAL, stock INT, category VARCHAR)",
        "question": "Liste todos os produtos que tem estoque maior que zero ordenados por preço"
    },
    {
        "category": "E-commerce",
        "name": "Total de vendas por produto",
        "difficulty": "Básico",
        "schema": "CREATE TABLE orders (id INT, product_id INT, quantity INT, total DECIMAL, order_date DATE)",
        "question": "Mostre o total de vendas e quantidade vendida por produto"
    },
    {
        "category": "E-commerce",
        "name": "Produtos mais vendidos",
        "difficulty": "Intermediário",
        "schema": "CREATE TABLE order_items (id INT, order_id INT, product_id INT, quantity INT, price DECIMAL); CREATE TABLE products (id INT, name VARCHAR, category VARCHAR)",
        "question": "Liste os 10 produtos mais vendidos com nome e categoria"
    },
    {
        "category": "E-commerce",
        "name": "Receita por categoria",
        "difficulty": "Intermediário",
        "schema": "CREATE TABLE products (id INT, name VARCHAR, category VARCHAR, price DECIMAL); CREATE TABLE order_items (order_id INT, product_id INT, quantity INT)",
        "question": "Calcule a receita total por categoria de produto"
    },
    {
        "category": "E-commerce",
        "name": "Pedidos do último mês",
        "difficulty": "Básico",
        "schema": "CREATE TABLE orders (id INT, customer_id INT, total DECIMAL, status VARCHAR, created_at DATE)",
        "question": "Liste todos os pedidos dos últimos 30 dias com status 'completed'"
    },
    
    # === CRM / CUSTOMERS ===
    {
        "category": "CRM",
        "name": "Clientes sem pedidos",
        "difficulty": "Intermediário",
        "schema": "CREATE TABLE customers (id INT, name VARCHAR, email VARCHAR, created_at DATE); CREATE TABLE orders (id INT, customer_id INT, total DECIMAL, order_date DATE)",
        "question": "Liste clientes que nunca fizeram nenhum pedido"
    },
    {
        "category": "CRM",
        "name": "Total gasto por cliente",
        "difficulty": "Básico",
        "schema": "CREATE TABLE customers (id INT, name VARCHAR, email VARCHAR); CREATE TABLE orders (id INT, customer_id INT, total DECIMAL)",
        "question": "Mostre o total gasto por cada cliente ordenado do maior para o menor"
    },
    {
        "category": "CRM",
        "name": "Clientes VIP",
        "difficulty": "Intermediário",
        "schema": "CREATE TABLE customers (id INT, name VARCHAR, email VARCHAR); CREATE TABLE orders (id INT, customer_id INT, total DECIMAL, order_date DATE)",
        "question": "Liste clientes que gastaram mais de 1000 reais no último ano"
    },
    {
        "category": "CRM",
        "name": "Novos clientes por mês",
        "difficulty": "Intermediário",
        "schema": "CREATE TABLE customers (id INT, name VARCHAR, email VARCHAR, created_at DATE)",
        "question": "Conte quantos novos clientes se cadastraram por mês em 2024"
    },
    
    # === ANALYTICS / DASHBOARDS ===
    {
        "category": "Analytics",
        "name": "Vendas por dia",
        "difficulty": "Básico",
        "schema": "CREATE TABLE orders (id INT, total DECIMAL, order_date DATE, status VARCHAR)",
        "question": "Mostre o total de vendas agrupado por dia da última semana"
    },
    {
        "category": "Analytics",
        "name": "Taxa de conversão",
        "difficulty": "Intermediário",
        "schema": "CREATE TABLE visits (id INT, user_id INT, page VARCHAR, visited_at TIMESTAMP); CREATE TABLE orders (id INT, user_id INT, total DECIMAL, created_at TIMESTAMP)",
        "question": "Calcule a taxa de conversão dividindo total de pedidos por total de visitas"
    },
    {
        "category": "Analytics",
        "name": "Ticket médio",
        "difficulty": "Básico",
        "schema": "CREATE TABLE orders (id INT, customer_id INT, total DECIMAL, order_date DATE)",
        "question": "Calcule o ticket médio dos pedidos do mês atual"
    },
    {
        "category": "Analytics",
        "name": "Produtos sem vendas",
        "difficulty": "Intermediário",
        "schema": "CREATE TABLE products (id INT, name VARCHAR, category VARCHAR, price DECIMAL); CREATE TABLE order_items (order_id INT, product_id INT, quantity INT)",
        "question": "Liste produtos que nunca foram vendidos"
    },
    
    # === FILTROS COMUNS ===
    {
        "category": "Filtros",
        "name": "Busca por nome",
        "difficulty": "Básico",
        "schema": "CREATE TABLE products (id INT, name VARCHAR, description TEXT, price DECIMAL)",
        "question": "Busque produtos que contenham 'notebook' no nome ou descrição"
    },
    {
        "category": "Filtros",
        "name": "Filtro por múltiplos valores",
        "difficulty": "Básico",
        "schema": "CREATE TABLE orders (id INT, status VARCHAR, customer_id INT, total DECIMAL)",
        "question": "Liste pedidos com status 'pending', 'processing' ou 'shipped'"
    },
    {
        "category": "Filtros",
        "name": "Filtro por faixa de preço",
        "difficulty": "Básico",
        "schema": "CREATE TABLE products (id INT, name VARCHAR, price DECIMAL, category VARCHAR)",
        "question": "Liste produtos com preço entre 100 e 500 reais"
    },
    {
        "category": "Filtros",
        "name": "Filtro por data",
        "difficulty": "Básico",
        "schema": "CREATE TABLE orders (id INT, customer_id INT, total DECIMAL, created_at DATE)",
        "question": "Liste pedidos criados entre 1 de janeiro e 31 de março de 2024"
    },
    
    # === RELATÓRIOS ===
    {
        "category": "Relatórios",
        "name": "Top 5 clientes",
        "difficulty": "Básico",
        "schema": "CREATE TABLE customers (id INT, name VARCHAR); CREATE TABLE orders (id INT, customer_id INT, total DECIMAL)",
        "question": "Liste os 5 clientes que mais gastaram com nome e total gasto"
    },
    {
        "category": "Relatórios",
        "name": "Produtos com baixo estoque",
        "difficulty": "Básico",
        "schema": "CREATE TABLE products (id INT, name VARCHAR, stock INT, min_stock INT, category VARCHAR)",
        "question": "Liste produtos onde o estoque atual está abaixo do estoque mínimo"
    },
    {
        "category": "Relatórios",
        "name": "Pedidos pendentes por cliente",
        "difficulty": "Intermediário",
        "schema": "CREATE TABLE customers (id INT, name VARCHAR, email VARCHAR); CREATE TABLE orders (id INT, customer_id INT, status VARCHAR, total DECIMAL, created_at DATE)",
        "question": "Liste clientes que tem pedidos com status 'pending' há mais de 7 dias"
    },
    
    # === JOINS BÁSICOS ===
    {
        "category": "Joins",
        "name": "Pedidos com dados do cliente",
        "difficulty": "Básico",
        "schema": "CREATE TABLE customers (id INT, name VARCHAR, email VARCHAR); CREATE TABLE orders (id INT, customer_id INT, total DECIMAL, order_date DATE)",
        "question": "Liste todos os pedidos com nome e email do cliente"
    },
    {
        "category": "Joins",
        "name": "Produtos e categorias",
        "difficulty": "Básico",
        "schema": "CREATE TABLE products (id INT, name VARCHAR, price DECIMAL, category_id INT); CREATE TABLE categories (id INT, name VARCHAR)",
        "question": "Liste todos os produtos com o nome da categoria"
    },
    {
        "category": "Joins",
        "name": "Usuários e seus últimos pedidos",
        "difficulty": "Intermediário",
        "schema": "CREATE TABLE users (id INT, name VARCHAR, email VARCHAR); CREATE TABLE orders (id INT, user_id INT, total DECIMAL, created_at DATE)",
        "question": "Para cada usuário mostre o valor do último pedido"
    },
    
    # === AGREGAÇÕES ===
    {
        "category": "Agregações",
        "name": "Contagem de pedidos por status",
        "difficulty": "Básico",
        "schema": "CREATE TABLE orders (id INT, status VARCHAR, total DECIMAL, created_at DATE)",
        "question": "Conte quantos pedidos existem para cada status"
    },
    {
        "category": "Agregações",
        "name": "Média de preço por categoria",
        "difficulty": "Básico",
        "schema": "CREATE TABLE products (id INT, name VARCHAR, price DECIMAL, category VARCHAR)",
        "question": "Calcule o preço médio dos produtos agrupados por categoria"
    },
    {
        "category": "Agregações",
        "name": "Min e Max por grupo",
        "difficulty": "Intermediário",
        "schema": "CREATE TABLE products (id INT, name VARCHAR, price DECIMAL, category VARCHAR)",
        "question": "Para cada categoria mostre o produto mais barato e mais caro"
    },
    
    # === CASOS PRÁTICOS ===
    {
        "category": "Prático",
        "name": "Produtos duplicados",
        "difficulty": "Intermediário",
        "schema": "CREATE TABLE products (id INT, name VARCHAR, sku VARCHAR, price DECIMAL)",
        "question": "Encontre produtos com o mesmo SKU (possíveis duplicatas)"
    },
    {
        "category": "Prático",
        "name": "Emails únicos",
        "difficulty": "Básico",
        "schema": "CREATE TABLE users (id INT, name VARCHAR, email VARCHAR, created_at DATE)",
        "question": "Liste todos os emails únicos de usuários cadastrados"
    },
    {
        "category": "Prático",
        "name": "Atualização de estoque",
        "difficulty": "Intermediário",
        "schema": "CREATE TABLE products (id INT, name VARCHAR, stock INT); CREATE TABLE order_items (order_id INT, product_id INT, quantity INT)",
        "question": "Mostre produtos e quanto de estoque foi vendido no total"
    },
    {
        "category": "Prático",
        "name": "Clientes inativos",
        "difficulty": "Intermediário",
        "schema": "CREATE TABLE customers (id INT, name VARCHAR, email VARCHAR); CREATE TABLE orders (id INT, customer_id INT, created_at DATE)",
        "question": "Liste clientes que não fazem pedidos há mais de 6 meses"
    }
]

print(f"\n{len(real_world_scenarios)} cenários de teste carregados\n")

# Gerar SQLs
results = []
categories_stats = {}

for i, scenario in enumerate(real_world_scenarios, 1):
    cat = scenario['category']
    if cat not in categories_stats:
        categories_stats[cat] = {"total": 0, "ok": 0}
    categories_stats[cat]["total"] += 1
    
    print(f"\n{'='*100}")
    print(f"[{i}/{len(real_world_scenarios)}] {scenario['category']} - {scenario['name']} ({scenario['difficulty']})")
    print(f"{'='*100}")
    print(f"\nSchema:")
    for line in scenario['schema'].split(';'):
        if line.strip():
            print(f"  {line.strip()}")
    print(f"\nPergunta: {scenario['question']}")
    print(f"\n{'-'*100}")
    
    messages = [
        {"role": "system", "content": f"Dialect: postgres\nSchema:\n{scenario['schema']}"},
        {"role": "user", "content": scenario['question']}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    gen_params = {
        "max_new_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    with torch.no_grad():
        output = model.generate(**inputs, **gen_params)
    
    sql = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Clean
    if ';' in sql:
        sql = sql.split(';')[0] + ';'
    sql = sql.replace('<think>\n', '').replace('<think>', '').strip()
    
    # Validate
    is_valid = sql.upper().startswith('SELECT') or 'SELECT' in sql.upper()[:50]
    
    if is_valid:
        categories_stats[cat]["ok"] += 1
        status = "[OK]"
    else:
        status = "[FAIL]"
    
    print(f"\n{status}")
    print(sql)
    
    results.append({
        'category': scenario['category'],
        'name': scenario['name'],
        'difficulty': scenario['difficulty'],
        'question': scenario['question'],
        'sql': sql,
        'valid': is_valid
    })

# Save results
print("\n" + "="*100)
print("SALVANDO RESULTADOS")
print("="*100)

output_file = "real_world_test_results.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("TESTE DE QUALIDADE - QUERIES SQL DO MUNDO REAL\n")
    f.write(f"Checkpoint: 1250\n")
    f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*100 + "\n\n")
    
    for i, r in enumerate(results, 1):
        f.write(f"\n{'='*100}\n")
        f.write(f"[{i}] {r['category']} - {r['name']} ({r['difficulty']})\n")
        f.write(f"{'='*100}\n")
        f.write(f"Pergunta: {r['question']}\n\n")
        f.write(f"SQL Gerado:\n{r['sql']}\n\n")
        f.write(f"Status: {'[OK] VALIDO' if r['valid'] else '[FAIL] INVALIDO'}\n")
        f.write(f"\nAVALIAÇÃO MANUAL:\n")
        f.write(f"  [ ] Sintaticamente correto\n")
        f.write(f"  [ ] Semanticamente correto (responde a pergunta)\n")
        f.write(f"  [ ] Otimizado (usa índices, evita subqueries desnecessárias)\n")
        f.write(f"  [ ] Produção-ready (pode usar em produção)\n")
        f.write(f"  Nota: ___/10\n")
        f.write(f"  Comentários: _________________________________\n")
        f.write(f"\n")

# Save JSON for programmatic analysis
json_output = "real_world_test_results.json"
with open(json_output, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nResultados salvos em:")
print(f"  - {output_file} (para revisão manual)")
print(f"  - {json_output} (para análise programática)")

# Print summary
print("\n" + "="*100)
print("RESUMO POR CATEGORIA")
print("="*100)

total_ok = sum(r['valid'] for r in results)
total_tests = len(results)

for cat, stats in sorted(categories_stats.items()):
    pct = (stats['ok'] / stats['total'] * 100) if stats['total'] > 0 else 0
    print(f"\n{cat:20s} {stats['ok']:2d}/{stats['total']:2d} ({pct:5.1f}%)")

print(f"\n{'='*100}")
print(f"TOTAL:              {total_ok:2d}/{total_tests:2d} ({total_ok/total_tests*100:5.1f}%)")
print(f"{'='*100}")

# Summary by difficulty
print("\n" + "="*100)
print("RESUMO POR DIFICULDADE")
print("="*100)

difficulty_stats = {}
for r in results:
    diff = r['difficulty']
    if diff not in difficulty_stats:
        difficulty_stats[diff] = {"total": 0, "ok": 0}
    difficulty_stats[diff]["total"] += 1
    if r['valid']:
        difficulty_stats[diff]["ok"] += 1

for diff, stats in sorted(difficulty_stats.items()):
    pct = (stats['ok'] / stats['total'] * 100) if stats['total'] > 0 else 0
    print(f"\n{diff:20s} {stats['ok']:2d}/{stats['total']:2d} ({pct:5.1f}%)")

print("\n" + "="*100)
print("TESTE COMPLETO!")
print("="*100)
print("\nPróximos passos:")
print("  1. Revise os resultados em real_world_test_results.txt")
print("  2. Preencha as avaliações manuais")
print("  3. Calcule a nota média")
print("  4. Se nota >= 8.0, checkpoint 1250 está pronto para package")

