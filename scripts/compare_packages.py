#!/usr/bin/env python3
"""
Compare the newly empacotado Expert-SQL (v0.3.0 checkpoint-500) com a versão anterior v0.2.1.

Para cada pacote (.expert):
  1. Extrai os arquivos para um diretório temporário
  2. Carrega o adapter PEFT e gera SQL para um conjunto fixo de prompts
  3. Avalia se o resultado parece SQL válido e se contém a palavra-chave esperada
"""
from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_PATH = Path("F:/Node/hivellm/expert/models/Qwen3-0.6B")
CURRENT_PACKAGE = Path("expert-sql-qwen3-0-6b.v0.3.0.expert")
PREVIOUS_PACKAGE = Path("expert-sql-qwen3-0-6b.v0.2.1.expert")

GEN_CONFIG = dict(
    max_new_tokens=120,
    do_sample=False,
    temperature=0.0,
    top_p=1.0,
    top_k=0,
    pad_token_id=0,  # will be set after tokenizer load
)

TEST_CASES: List[Dict[str, str]] = [
    {
        "cat": "SELECT",
        "schema": "CREATE TABLE users (id INT, name VARCHAR, email VARCHAR)",
        "question": "Liste todos os usuarios",
    },
    {
        "cat": "WHERE",
        "schema": "CREATE TABLE products (id INT, name VARCHAR, price DECIMAL, stock INT)",
        "question": "Mostre produtos com preco menor que 100",
    },
    {
        "cat": "COUNT",
        "schema": "CREATE TABLE orders (id INT, status VARCHAR, total DECIMAL)",
        "question": "Quantos pedidos foram cancelados?",
    },
    {
        "cat": "JOIN",
        "schema": "CREATE TABLE customers (id INT, name VARCHAR); CREATE TABLE orders (id INT, customer_id INT, amount DECIMAL)",
        "question": "Liste clientes e seus pedidos",
    },
    {
        "cat": "GROUP BY",
        "schema": "CREATE TABLE sales (product VARCHAR, quantity INT, revenue DECIMAL)",
        "question": "Qual a receita total por produto?",
    },
    {
        "cat": "HAVING",
        "schema": "CREATE TABLE transactions (id INT, account INT, amount DECIMAL, type VARCHAR)",
        "question": "Contas com total de transacoes acima de 10000",
    },
    {
        "cat": "UNION",
        "schema": "CREATE TABLE customers_br (id INT, name VARCHAR); CREATE TABLE customers_us (id INT, name VARCHAR)",
        "question": "Todos os clientes do Brasil e EUA",
    },
    {
        "cat": "LEFT JOIN",
        "schema": "CREATE TABLE authors (id INT, name VARCHAR); CREATE TABLE books (id INT, author_id INT, title VARCHAR)",
        "question": "Autores e seus livros, incluindo autores sem livros",
    },
    {
        "cat": "CASE",
        "schema": "CREATE TABLE products (id INT, name VARCHAR, price DECIMAL, category VARCHAR)",
        "question": "Classifique produtos como 'Caro' se preco > 1000, senao 'Barato'",
    },
]


def extract_package(package_path: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="expert_sql_pkg_"))
    with tarfile.open(package_path, "r:gz") as tar:
        tar.extractall(temp_dir)
    return temp_dir


def load_model_with_adapter(adapter_dir: Path):
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_PATH),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    return model


def iter_cases(tokenizer):
    for case in TEST_CASES:
        messages = [
            {"role": "system", "content": f"Dialect: postgres\nSchema:\n{case['schema']}"},
            {"role": "user", "content": case["question"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to("cuda")
        yield case, inputs


def clean_sql(raw: str) -> str:
    sql = raw.replace("<think>", "").replace("</think>", "").strip()
    if ";" in sql:
        sql = sql.split(";", 1)[0] + ";"
    return sql.strip()


def evaluate_sql(sql: str, category: str) -> Dict[str, bool]:
    upper = sql.upper()
    is_sql = "SELECT" in upper and "FROM" in upper and "EXPLAIN" not in upper
    has_keyword = category.upper() in upper or category.lower() in sql.lower()
    return {"is_sql": is_sql, "has_keyword": has_keyword}


def run_package(package_path: Path, label: str, tokenizer) -> Dict[str, int]:
    extract_dir = extract_package(package_path)
    adapter_dir = extract_dir / "qwen3-06b" / "checkpoint-500"
    if not adapter_dir.exists():
        adapter_dir = extract_dir

    model = load_model_with_adapter(adapter_dir)
    model.eval()

    sql_hits = 0
    keyword_hits = 0
    details = []

    with torch.inference_mode():
        for case, inputs in iter_cases(tokenizer):
            outputs = model.generate(
                **inputs,
                max_new_tokens=GEN_CONFIG["max_new_tokens"],
                do_sample=GEN_CONFIG["do_sample"],
                temperature=GEN_CONFIG["temperature"],
                top_p=GEN_CONFIG["top_p"],
                top_k=GEN_CONFIG["top_k"],
                pad_token_id=tokenizer.eos_token_id,
            )
            raw_sql = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            formatted = clean_sql(raw_sql)
            evaluation = evaluate_sql(formatted, case["cat"])
            sql_hits += int(evaluation["is_sql"])
            keyword_hits += int(evaluation["has_keyword"])
            details.append((case["cat"], formatted, evaluation))

    del model
    torch.cuda.empty_cache()
    shutil.rmtree(extract_dir, ignore_errors=True)

    return {"label": label, "sql_hits": sql_hits, "keyword_hits": keyword_hits, "details": details}


def main() -> None:
    if not CURRENT_PACKAGE.exists():
        raise FileNotFoundError(f"Pacote atual não encontrado: {CURRENT_PACKAGE}")
    if not PREVIOUS_PACKAGE.exists():
        raise FileNotFoundError(f"Pacote anterior não encontrado: {PREVIOUS_PACKAGE}")

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_PATH), trust_remote_code=True)

    print("=" * 80)
    print("Comparação de Pacotes: v0.3.0 (checkpoint-500) vs v0.2.1")
    print("=" * 80)

    current = run_package(CURRENT_PACKAGE, "v0.3.0 (ckpt-500)", tokenizer)
    previous = run_package(PREVIOUS_PACKAGE, "v0.2.1", tokenizer)

    total = len(TEST_CASES)
    print("\nResumo:")
    print(f"{current['label']:<25} SQL válido: {current['sql_hits']:>2}/{total} | keyword: {current['keyword_hits']:>2}/{total}")
    print(f"{previous['label']:<25} SQL válido: {previous['sql_hits']:>2}/{total} | keyword: {previous['keyword_hits']:>2}/{total}")

    print("\nDetalhes (v0.3.0):")
    for cat, sql, eval_res in current["details"]:
        print(f"[{cat}] SQL? {'SIM' if eval_res['is_sql'] else 'NAO'} | KW? {'SIM' if eval_res['has_keyword'] else 'NAO'} -> {sql}")

    print("\nDiferenças vs v0.2.1:")
    for idx, (case, curr_detail) in enumerate(zip(TEST_CASES, current["details"])):
        prev_detail = previous["details"][idx]
        curr_eval = curr_detail[2]
        prev_eval = prev_detail[2]
        if curr_eval["is_sql"] and not prev_eval["is_sql"]:
            print(f"- {case['cat']}: agora gera SQL válido (antes, v0.2.1 não gerava).")
        elif curr_eval["has_keyword"] and not prev_eval["has_keyword"]:
            print(f"- {case['cat']}: agora inclui palavra-chave esperada (antes não).")

    print("\nFim da comparação.")


if __name__ == "__main__":
    main()

