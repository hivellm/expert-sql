"""Análise Qualitativa Profunda por Especialista SQL - Cenários Complexos"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_PATH = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
CHECKPOINT_750 = "weights/qwen3-06b/checkpoint-750"
CHECKPOINT_1000 = "weights/qwen3-06b/checkpoint-1000"
CHECKPOINT_1250 = "weights/qwen3-06b/checkpoint-1250"
CHECKPOINT_1500 = "weights/qwen3-06b/checkpoint-1500"

print("=" * 100)
print("ANALISE QUALITATIVA PROFUNDA - CENARIOS SQL COMPLEXOS")
print("Analise por: Claude (Especialista SQL)")
print("=" * 100)

print("\nCarregando modelos...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

print("Carregando checkpoint 750...")
model_750 = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)
model_750 = PeftModel.from_pretrained(model_750, CHECKPOINT_750)

print("Carregando checkpoint 1000...")
model_1000 = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)
model_1000 = PeftModel.from_pretrained(model_1000, CHECKPOINT_1000)

print("Carregando checkpoint 1250...")
model_1250 = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)
model_1250 = PeftModel.from_pretrained(model_1250, CHECKPOINT_1250)

print("Carregando checkpoint 1500...")
model_1500 = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)
model_1500 = PeftModel.from_pretrained(model_1500, CHECKPOINT_1500)

# Cenários SQL COMPLEXOS
complex_scenarios = [
    {
        "name": "JOIN Multiplo com Agregacao",
        "schema": """CREATE TABLE customers (id INT, name VARCHAR, country VARCHAR);
CREATE TABLE orders (id INT, customer_id INT, order_date DATE, total DECIMAL);
CREATE TABLE order_items (id INT, order_id INT, product_id INT, quantity INT, price DECIMAL);
CREATE TABLE products (id INT, name VARCHAR, category VARCHAR)""",
        "question": "Mostre o valor total de pedidos por categoria de produto para clientes do Brasil em 2024"
    },
    {
        "name": "Subquery Correlacionada",
        "schema": """CREATE TABLE employees (id INT, name VARCHAR, salary DECIMAL, department_id INT);
CREATE TABLE departments (id INT, name VARCHAR, budget DECIMAL)""",
        "question": "Liste funcionarios cujo salario e maior que a media do seu proprio departamento"
    },
    {
        "name": "Window Function",
        "schema": """CREATE TABLE sales (id INT, salesperson VARCHAR, amount DECIMAL, sale_date DATE, region VARCHAR)""",
        "question": "Ranking dos vendedores por valor total de vendas em cada regiao"
    },
    {
        "name": "CTE Recursiva",
        "schema": """CREATE TABLE employees (id INT, name VARCHAR, manager_id INT, salary DECIMAL)""",
        "question": "Mostre a hierarquia completa de funcionarios com seus gerentes"
    },
    {
        "name": "UNION com Agregacoes Diferentes",
        "schema": """CREATE TABLE sales_q1 (product VARCHAR, revenue DECIMAL);
CREATE TABLE sales_q2 (product VARCHAR, revenue DECIMAL);
CREATE TABLE sales_q3 (product VARCHAR, revenue DECIMAL);
CREATE TABLE sales_q4 (product VARCHAR, revenue DECIMAL)""",
        "question": "Consolide todas as vendas do ano e mostre o total por produto"
    },
    {
        "name": "Subquery em SELECT e WHERE",
        "schema": """CREATE TABLE products (id INT, name VARCHAR, price DECIMAL, category_id INT);
CREATE TABLE categories (id INT, name VARCHAR);
CREATE TABLE sales (product_id INT, quantity INT, sale_date DATE)""",
        "question": "Para cada produto, mostre nome, categoria e percentual das vendas totais da sua categoria"
    },
    {
        "name": "HAVING com Agregacao Complexa",
        "schema": """CREATE TABLE transactions (id INT, account_id INT, amount DECIMAL, type VARCHAR, date DATE)""",
        "question": "Contas que tiveram mais de 10 transacoes e saldo final positivo no mes"
    },
    {
        "name": "LEFT JOIN com Condicoes Multiplas",
        "schema": """CREATE TABLE customers (id INT, name VARCHAR, segment VARCHAR);
CREATE TABLE orders (id INT, customer_id INT, status VARCHAR, total DECIMAL);
CREATE TABLE payments (order_id INT, amount DECIMAL, payment_date DATE)""",
        "question": "Clientes premium com pedidos confirmados mas sem pagamento recebido"
    },
    {
        "name": "CASE WHEN Aninhado",
        "schema": """CREATE TABLE employees (id INT, name VARCHAR, salary DECIMAL, years_service INT, performance_score DECIMAL)""",
        "question": "Classifique funcionarios em Junior/Pleno/Senior baseado em salario E anos de servico, com bonus se performance > 8"
    },
    {
        "name": "EXISTS vs IN Optimization",
        "schema": """CREATE TABLE authors (id INT, name VARCHAR, country VARCHAR);
CREATE TABLE books (id INT, author_id INT, title VARCHAR, year INT);
CREATE TABLE awards (book_id INT, award_name VARCHAR, year INT)""",
        "question": "Autores brasileiros que tem pelo menos um livro premiado depois de 2020"
    }
]

print("\nGerando SQLs para analise qualitativa...\n")

results = []

for i, scenario in enumerate(complex_scenarios, 1):
    print(f"\n{'='*100}")
    print(f"CENARIO {i}: {scenario['name']}")
    print(f"{'='*100}")
    print(f"\nSchema:")
    for line in scenario['schema'].split('\n'):
        if line.strip():
            print(f"  {line.strip()}")
    print(f"\nPergunta: {scenario['question']}")
    print(f"\n{'-'*100}")
    
    messages = [
        {"role": "system", "content": f"Dialect: postgres\nSchema:\n{scenario['schema']}"},
        {"role": "user", "content": scenario['question']}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(base_model.device)
    
    gen_params = {
        "max_new_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    # Generate
    with torch.no_grad():
        out_base = base_model.generate(**inputs, **gen_params)
        out_750 = model_750.generate(**inputs, **gen_params)
        out_1000 = model_1000.generate(**inputs, **gen_params)
        out_1250 = model_1250.generate(**inputs, **gen_params)
        out_1500 = model_1500.generate(**inputs, **gen_params)
    
    sql_base = tokenizer.decode(out_base[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    sql_750 = tokenizer.decode(out_750[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    sql_1000 = tokenizer.decode(out_1000[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    sql_1250 = tokenizer.decode(out_1250[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    sql_1500 = tokenizer.decode(out_1500[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Clean
    for sql_var in ['sql_base', 'sql_750', 'sql_1000', 'sql_1250', 'sql_1500']:
        sql = locals()[sql_var]
        if ';' in sql:
            sql = sql.split(';')[0] + ';'
        sql = sql.replace('<think>\n', '').replace('<think>', '').strip()
        locals()[sql_var] = sql
    
    print("\n[BASE MODEL]")
    print(sql_base)
    
    print("\n[CHECKPOINT-750]")
    print(sql_750)
    
    print("\n[CHECKPOINT-1000]")
    print(sql_1000)
    
    print("\n[CHECKPOINT-1250]")
    print(sql_1250)
    
    print("\n[CHECKPOINT-1500]")
    print(sql_1500)
    
    results.append({
        'name': scenario['name'],
        'question': scenario['question'],
        'sql_base': sql_base,
        'sql_750': sql_750,
        'sql_1000': sql_1000,
        'sql_1250': sql_1250,
        'sql_1500': sql_1500
    })

# Save results for manual analysis
print("\n" + "="*100)
print("RESULTADOS SALVOS - AGUARDANDO ANALISE QUALITATIVA")
print("="*100)

output_file = "sql_analysis_results.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("ANALISE QUALITATIVA PROFUNDA - CENARIOS SQL COMPLEXOS\n")
    f.write("="*100 + "\n\n")
    
    for i, r in enumerate(results, 1):
        f.write(f"\nCENARIO {i}: {r['name']}\n")
        f.write("-"*100 + "\n")
        f.write(f"Pergunta: {r['question']}\n\n")
        
        f.write("BASE MODEL:\n")
        f.write(r['sql_base'] + "\n\n")
        
        f.write("CHECKPOINT-750:\n")
        f.write(r['sql_750'] + "\n\n")
        
        f.write("CHECKPOINT-1000:\n")
        f.write(r['sql_1000'] + "\n\n")
        
        f.write("CHECKPOINT-1250:\n")
        f.write(r['sql_1250'] + "\n\n")
        
        f.write("CHECKPOINT-1500:\n")
        f.write(r['sql_1500'] + "\n\n")
        
        f.write("ANALISE QUALITATIVA (para ser preenchida):\n")
        f.write("  Corretude (BASE):     [  ]/10\n")
        f.write("  Corretude (750):      [  ]/10\n")
        f.write("  Corretude (1000):     [  ]/10\n")
        f.write("  Corretude (1250):     [  ]/10\n")
        f.write("  Corretude (1500):     [  ]/10\n")
        f.write("  Otimizacao (BASE):    [  ]/10\n")
        f.write("  Otimizacao (750):     [  ]/10\n")
        f.write("  Otimizacao (1000):    [  ]/10\n")
        f.write("  Otimizacao (1250):    [  ]/10\n")
        f.write("  Otimizacao (1500):    [  ]/10\n")
        f.write("  Completude (BASE):    [  ]/10\n")
        f.write("  Completude (750):     [  ]/10\n")
        f.write("  Completude (1000):    [  ]/10\n")
        f.write("  Completude (1250):    [  ]/10\n")
        f.write("  Completude (1500):    [  ]/10\n")
        f.write("  Melhor solucao:       [           ]\n")
        f.write("  Comentarios:          \n\n")
        f.write("="*100 + "\n\n")

print(f"\nResultados salvos em: {output_file}")
print("\nPor favor, analise cada SQL e forneça:")
print("  1. Corretude: O SQL esta sintaticamente correto e responde a pergunta?")
print("  2. Otimizacao: O SQL usa a melhor estrategia (JOINs vs subqueries, indices, etc)?")
print("  3. Completude: O SQL cobre todos os requisitos da pergunta?")
print("  4. Melhor solucao: Qual modelo gerou o melhor SQL?")
print("\n" + "="*100)

# Print summary table
print("\n\nTABELA PARA ANALISE MANUAL:")
print("="*120)
print(f"{'#':<3} {'CENARIO':<30} {'BASE':<8} {'750':<8} {'1000':<8} {'1250':<8} {'1500':<8} {'EVOLUCAO':<15}")
print("-"*120)
for i, r in enumerate(results, 1):
    # Check if SQL starts with SELECT (basic validation)
    base_ok = "OK" if r['sql_base'].upper().startswith('SELECT') or 'SELECT' in r['sql_base'].upper()[:50] else "FAIL"
    ckp750_ok = "OK" if r['sql_750'].upper().startswith('SELECT') or 'SELECT' in r['sql_750'].upper()[:50] else "FAIL"
    ckp1000_ok = "OK" if r['sql_1000'].upper().startswith('SELECT') or 'SELECT' in r['sql_1000'].upper()[:50] else "FAIL"
    ckp1250_ok = "OK" if r['sql_1250'].upper().startswith('SELECT') or 'SELECT' in r['sql_1250'].upper()[:50] else "FAIL"
    ckp1500_ok = "OK" if r['sql_1500'].upper().startswith('SELECT') or 'SELECT' in r['sql_1500'].upper()[:50] else "FAIL"
    
    print(f"{i:<3} {r['name']:<30} {base_ok:<8} {ckp750_ok:<8} {ckp1000_ok:<8} {ckp1250_ok:<8} {ckp1500_ok:<8} {'[ANALISAR]':<15}")

print("="*120)
print(f"\nArquivo completo: {output_file}")
print("Analise cada SQL manualmente e preencha as notas!")
print("\nLEGENDA:")
print("  OK   = SQL sintaticamente valido (inicia com SELECT)")
print("  FAIL = SQL invalido ou incompleto")
print("  EVOLUCAO = Compara checkpoint 750 -> 1500 para ver se melhorou ou piorou")

