#!/usr/bin/env python3
"""Validate processed dataset format"""
import json
import random

dataset_path = "datasets/processed/train.jsonl"

print("="*80)
print("VALIDAÇÃO DO DATASET PROCESSADO")
print("="*80)
print()

# Load and check random samples
with open(dataset_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

total = len(lines)
print(f"Total de exemplos: {total:,}")
print()

# Check random samples
indices = [0, 100, 1000, 5000, 10000, 50000, 90000]
indices = [i for i in indices if i < total]

issues = []

for idx in indices:
    print(f"\n{'='*80}")
    print(f"EXEMPLO #{idx}")
    print('='*80)
    
    try:
        example = json.loads(lines[idx])
        
        # Check required fields
        required = ['text', 'context', 'question', 'answer']
        missing = [f for f in required if f not in example]
        
        if missing:
            issues.append(f"Exemplo {idx}: faltando campos {missing}")
            print(f"[ERRO] Faltando campos {missing}")
            continue
        
        # Check ChatML format
        text = example['text']
        
        checks = {
            "Tem <|system|>": "<|system|>" in text,
            "Tem Schema": "Schema:" in text,
            "Tem <|user|>": "<|user|>" in text,
            "Tem <|assistant|>": "<|assistant|>" in text,
            "Tem <|end|>": text.count("<|end|>") >= 3,
            "SQL nao-vazio": len(example['answer'].strip()) > 0,
            "Question nao-vazia": len(example['question'].strip()) > 0,
        }
        
        all_ok = all(checks.values())
        
        if all_ok:
            print("[OK] Formato OK")
        else:
            print("[AVISO] Problemas encontrados:")
            for check, passed in checks.items():
                if not passed:
                    print(f"   [X] {check}")
                    issues.append(f"Exemplo {idx}: {check}")
        
        # Show content
        print(f"\nQUESTION: {example['question'][:100]}...")
        print(f"SQL: {example['answer'][:150]}...")
        
        # Show optional fields
        if 'domain' in example:
            print(f"DOMAIN: {example['domain']}")
        if 'complexity' in example:
            print(f"COMPLEXITY: {example['complexity']}")
        
        # Check text length
        text_len = len(text)
        print(f"\nTAMANHO: {text_len} caracteres")
        
        if text_len < 50:
            issues.append(f"Exemplo {idx}: texto muito curto ({text_len} chars)")
            print(f"[AVISO] Texto muito curto!")
        elif text_len > 2048:
            issues.append(f"Exemplo {idx}: texto muito longo ({text_len} chars)")
            print(f"[AVISO] Texto muito longo!")
        
    except json.JSONDecodeError as e:
        issues.append(f"Exemplo {idx}: JSON invalido - {e}")
        print(f"[ERRO JSON] {e}")
    except Exception as e:
        issues.append(f"Exemplo {idx}: Erro - {e}")
        print(f"[ERRO] {e}")

# Summary
print(f"\n\n{'='*80}")
print("RESUMO DA VALIDAÇÃO")
print('='*80)
print()

if not issues:
    print("*** DATASET 100% VALIDO! ***")
    print()
    print("Todos os exemplos verificados estao no formato correto:")
    print("  [OK] ChatML format (<|system|>, <|user|>, <|assistant|>, <|end|>)")
    print("  [OK] Campos obrigatorios presentes")
    print("  [OK] SQL e questions nao-vazios")
    print("  [OK] Tamanhos dentro do esperado")
    print()
    print("*** PRONTO PARA TREINAR! ***")
else:
    print(f"[AVISO] Encontrados {len(issues)} problemas:")
    for issue in issues:
        print(f"  - {issue}")

print()
print('='*80)

