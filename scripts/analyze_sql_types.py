#!/usr/bin/env python3
"""Analyze SQL command types in dataset"""
import json
from collections import Counter

dataset_path = "datasets/processed/train.jsonl"

print("="*80)
print("ANÁLISE DE TIPOS DE COMANDOS SQL")
print("="*80)
print()

commands = Counter()
task_types = Counter()
complexity = Counter()

with open(dataset_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        example = json.loads(line)
        sql = example['answer'].strip().upper()
        
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
        elif sql.startswith('WITH'):
            commands['WITH (CTE)'] += 1
        else:
            commands['OTHER'] += 1
        
        # Task types
        if 'sql_task_type' in example:
            task_types[example['sql_task_type']] += 1
        
        # Complexity
        if 'complexity' in example:
            complexity[example['complexity']] += 1
        
        if (i + 1) % 10000 == 0:
            print(f"Analisado: {i+1:,} exemplos...")

total = sum(commands.values())

print()
print("="*80)
print("TIPOS DE COMANDOS SQL")
print("="*80)
for cmd, count in commands.most_common():
    pct = count / total * 100
    print(f"{cmd:15} {count:6,} ({pct:5.1f}%)")

print()
print("="*80)
print("TIPOS DE TAREFAS")
print("="*80)
for task, count in task_types.most_common():
    pct = count / total * 100
    print(f"{task:30} {count:6,} ({pct:5.1f}%)")

print()
print("="*80)
print("COMPLEXIDADE")
print("="*80)
for comp, count in complexity.most_common():
    pct = count / total * 100
    print(f"{comp:20} {count:6,} ({pct:5.1f}%)")

print()
print("="*80)
print("RESUMO")
print("="*80)
select_pct = commands['SELECT'] / total * 100
write_pct = (commands['INSERT'] + commands['UPDATE'] + commands['DELETE']) / total * 100
ddl_pct = (commands['CREATE'] + commands['DROP'] + commands['ALTER']) / total * 100

print(f"Total exemplos: {total:,}")
print(f"SELECT (leitura): {commands['SELECT']:,} ({select_pct:.1f}%)")
print(f"INSERT/UPDATE/DELETE (escrita): {commands['INSERT'] + commands['UPDATE'] + commands['DELETE']:,} ({write_pct:.1f}%)")
print(f"CREATE/DROP/ALTER (DDL): {commands['CREATE'] + commands['DROP'] + commands['ALTER']:,} ({ddl_pct:.1f}%)")
print()

if select_pct > 90:
    print("⚠️  PROBLEMA: Dataset é QUASE SÓ SELECT!")
    print("   Faltam comandos de escrita (INSERT, UPDATE, DELETE)")
    print("   Faltam comandos DDL (CREATE, ALTER, DROP)")
elif select_pct > 70:
    print("⚠️  Dataset tem MUITO SELECT, mas tem alguma diversidade")
else:
    print("✅ Dataset tem boa diversidade de comandos")

print("="*80)

