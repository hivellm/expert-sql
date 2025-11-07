#!/usr/bin/env python3
"""Verify MySQL->PostgreSQL fixes in dataset"""
import json
import re

dataset_path = "datasets/processed/train.jsonl"

print("="*80)
print("VERIFICANDO CORREÇÕES MySQL->PostgreSQL")
print("="*80)
print()

mysql_functions = {
    'YEAR(': 0,
    'MONTH(': 0,
    'DAY(': 0,
    'DATE_SUB': 0,
    'STR_TO_DATE': 0
}

postgres_functions = {
    'EXTRACT(YEAR': 0,
    'EXTRACT(MONTH': 0,
    'EXTRACT(DAY': 0,
    'INTERVAL': 0,
    'TO_DATE': 0
}

samples_with_extract = []

with open(dataset_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        example = json.loads(line)
        sql = example['answer'].upper()
        
        # Check MySQL functions (should be 0)
        for func in mysql_functions:
            if func in sql:
                mysql_functions[func] += 1
        
        # Check PostgreSQL functions (should exist)
        for func in postgres_functions:
            if func in sql:
                postgres_functions[func] += 1
                if 'EXTRACT' in func and len(samples_with_extract) < 5:
                    samples_with_extract.append({
                        'index': i,
                        'question': example['question'],
                        'sql': example['answer']
                    })

print("MySQL functions encontradas (RUIM se > 0):")
for func, count in mysql_functions.items():
    status = "[OK]" if count == 0 else "[PROBLEMA]"
    print(f"  {status} {func:15} {count:6,}")

print()
print("PostgreSQL functions encontradas (BOM se > 0):")
for func, count in postgres_functions.items():
    status = "[OK]" if count > 0 else "[INFO]"
    print(f"  {status} {func:15} {count:6,}")

print()
print("="*80)
print("EXEMPLOS COM EXTRACT (PostgreSQL correto):")
print("="*80)
for sample in samples_with_extract[:3]:
    print(f"\n[Exemplo #{sample['index']}]")
    print(f"Question: {sample['question'][:80]}...")
    print(f"SQL: {sample['sql'][:120]}...")

print()
print("="*80)
if sum(mysql_functions.values()) == 0:
    print("[OK] NENHUMA funcao MySQL encontrada!")
    print("     Todas as funcoes foram convertidas para PostgreSQL")
else:
    print(f"[AVISO] {sum(mysql_functions.values())} funcoes MySQL ainda presentes")
    print("        Validacao pode ter falhado em alguns casos")

print("="*80)

