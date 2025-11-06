#!/usr/bin/env python3
"""Download and analyze Spider dataset"""
from datasets import load_dataset
from collections import Counter
import json

print("="*80)
print("BAIXANDO E ANALISANDO SPIDER DATASET")
print("="*80)
print()

print("Carregando xlangai/spider...")
dataset = load_dataset("xlangai/spider", split="train")
print(f"Total de exemplos: {len(dataset)}")
print()

commands = Counter()
has_join = 0
has_subquery = 0
has_union = 0
has_group_by = 0
has_order_by = 0
has_limit = 0

print("Analisando comandos SQL...")
for example in dataset:
    sql = example['query'].strip().upper()
    
    # Detect command type
    if sql.startswith('SELECT'):
        commands['SELECT'] += 1
    elif sql.startswith('INSERT'):
        commands['INSERT'] += 1
    elif sql.startswith('UPDATE'):
        commands['UPDATE'] += 1
    elif sql.startswith('DELETE'):
        commands['DELETE'] += 1
    elif sql.startswith('CREATE'):
        commands['CREATE'] += 1
    elif sql.startswith('DROP'):
        commands['DROP'] += 1
    elif sql.startswith('ALTER'):
        commands['ALTER'] += 1
    else:
        commands['OTHER'] += 1
    
    # Check patterns
    if ' JOIN ' in sql:
        has_join += 1
    if 'SELECT' in sql and sql.count('SELECT') > 1:
        has_subquery += 1
    if ' UNION ' in sql:
        has_union += 1
    if ' GROUP BY ' in sql:
        has_group_by += 1
    if ' ORDER BY ' in sql:
        has_order_by += 1
    if ' LIMIT ' in sql:
        has_limit += 1

total = len(dataset)

print()
print("="*80)
print("TIPOS DE COMANDOS SQL")
print("="*80)
for cmd, count in commands.most_common():
    pct = count / total * 100
    print(f"{cmd:15} {count:6,} ({pct:5.1f}%)")

print()
print("="*80)
print("PADRÕES SQL")
print("="*80)
print(f"JOIN:       {has_join:6,} ({has_join/total*100:5.1f}%)")
print(f"Subquery:   {has_subquery:6,} ({has_subquery/total*100:5.1f}%)")
print(f"UNION:      {has_union:6,} ({has_union/total*100:5.1f}%)")
print(f"GROUP BY:   {has_group_by:6,} ({has_group_by/total*100:5.1f}%)")
print(f"ORDER BY:   {has_order_by:6,} ({has_order_by/total*100:5.1f}%)")
print(f"LIMIT:      {has_limit:6,} ({has_limit/total*100:5.1f}%)")

print()
print("="*80)
print("AMOSTRAS")
print("="*80)

# Show some examples
for i in [0, 100, 500, 1000, 2000]:
    if i < total:
        ex = dataset[i]
        print(f"\nExemplo #{i}:")
        print(f"Question: {ex['question'][:80]}...")
        print(f"SQL: {ex['query'][:120]}...")

print()
print("="*80)
print("COMPARAÇÃO: Spider vs Gretelai")
print("="*80)

# Spider stats
spider_select = commands['SELECT'] / total * 100
spider_write = (commands['INSERT'] + commands['UPDATE'] + commands['DELETE']) / total * 100
spider_ddl = (commands['CREATE'] + commands['DROP'] + commands['ALTER']) / total * 100

print("\nSPIDER:")
print(f"  SELECT: {spider_select:.1f}%")
print(f"  INSERT/UPDATE/DELETE: {spider_write:.1f}%")
print(f"  CREATE/DROP/ALTER: {spider_ddl:.1f}%")

print("\nGRETELAI (do arquivo):")
print(f"  SELECT: 89.5%")
print(f"  INSERT/UPDATE/DELETE: 9.3%")
print(f"  CREATE/DROP/ALTER: 0.8%")

print()
if spider_select > 95:
    print("[RESULTADO] Spider tambem e QUASE SO SELECT!")
    print("            Nao e mais balanceado que gretelai")
elif spider_select > 80:
    print("[RESULTADO] Spider tem um pouco mais de balanceamento")
    print("            Mas ainda e majoritariamente SELECT")
else:
    print("[RESULTADO] Spider e mais balanceado!")
    print("            Tem boa diversidade de comandos")

print("="*80)

