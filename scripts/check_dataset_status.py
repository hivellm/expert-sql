#!/usr/bin/env python3
"""Verifica se o dataset do SQL está pronto para treinar"""

import json
import re
from pathlib import Path

def main():
    dataset_path = Path("datasets/train.jsonl")
    
    if not dataset_path.exists():
        print("ERRO: Dataset não encontrado!")
        return
    
    print("="*80)
    print("VERIFICACAO DO DATASET SQL")
    print("="*80)
    
    total = 0
    with_reasoning = 0
    qwen3_format = 0
    direct_sql = 0
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1000:  # Verificar primeiros 1000
                break
            
            total += 1
            try:
                data = json.loads(line)
                text = data.get('text', '')
                
                # Verificar formato Qwen3
                if '<|im_start|>' in text:
                    qwen3_format += 1
                
                # Extrair resposta do assistant
                assistant_match = re.search(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', text, re.DOTALL)
                if assistant_match:
                    response = assistant_match.group(1).strip()
                    
                    # Verificar se tem reasoning block
                    if '<think>' in response or '<think>' in response:
                        with_reasoning += 1
                    elif response.startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE')):
                        direct_sql += 1
                else:
                    # Se não encontrou formato Qwen3, pode ser formato antigo
                    pass
                    
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON on line {i+1}")
    
    print(f"\nTotal analisados: {total}")
    print(f"Formato Qwen3: {qwen3_format} ({qwen3_format/total*100:.1f}%)")
    print(f"Com reasoning blocks: {with_reasoning} ({with_reasoning/total*100:.1f}%)")
    print(f"SQL direto (sem reasoning): {direct_sql} ({direct_sql/total*100:.1f}%)")
    
    print("\n" + "="*80)
    print("AVALIACAO")
    print("="*80)
    
    reasoning_rate = with_reasoning / total if total > 0 else 0
    qwen3_rate = qwen3_format / total if total > 0 else 0
    
    # Qwen3 training notebook usa 75% reasoning + 25% direct
    if qwen3_rate >= 0.95:
        print("[OK] Formato Qwen3 correto")
    else:
        print(f"[AVISO] Formato Qwen3: {qwen3_rate*100:.1f}% (esperado: 95%+)")
        print("        Dataset pode precisar ser regenerado")
    
    if 0.70 <= reasoning_rate <= 0.80:
        print("[OK] Distribuicao de reasoning correta (70-80%)")
        print("     Dataset pronto para treinamento com Qwen3")
    elif reasoning_rate < 0.70:
        print(f"[AVISO] Poucos exemplos com reasoning: {reasoning_rate*100:.1f}%")
        print("        Esperado: 70-80%")
        print("        Dataset precisa ser regenerado")
    else:
        print(f"[AVISO] Muitos exemplos com reasoning: {reasoning_rate*100:.1f}%")
        print("        Esperado: 70-80%")
        print("        Dataset pode precisar ser regenerado")
    
    print("\n" + "="*80)
    
    # Veredito final
    if qwen3_rate >= 0.95 and 0.70 <= reasoning_rate <= 0.80:
        print("\n[VEREDITO] SIM - Dataset pronto para treinar!")
    else:
        print("\n[VEREDITO] NAO - Dataset precisa ser regenerado")
        print("           Execute: python preprocess.py --output datasets --format chatml")

if __name__ == "__main__":
    main()

