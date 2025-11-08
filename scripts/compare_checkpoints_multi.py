#!/usr/bin/env python3
"""
Compare SQL generation quality across available checkpoints.

Evaluates the base model and each adapter on a shared set of prompts and reports:
- Whether the output looks like executable SQL (SELECT + FROM and no plain-text explanation)
- Whether the expected keyword/category appears in the SQL
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL_PATH = Path("F:/Node/hivellm/expert/models/Qwen3-0.6B")
CHECKPOINTS: Dict[str, Path | None] = {
    "base": None,
    "checkpoint-500": Path("weights/qwen3-06b/checkpoint-500"),
    "checkpoint-750": Path("weights/qwen3-06b/checkpoint-750"),
    "checkpoint-770": Path("weights/qwen3-06b/checkpoint-770"),
    "final": Path("weights/qwen3-06b/final"),
}

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
        "schema": "CREATE TABLE orders (id INT, user_id INT, total DECIMAL, status VARCHAR)",
        "question": "Quantos pedidos foram cancelados?",
    },
    {
        "cat": "JOIN",
        "schema": "CREATE TABLE customers (id INT, name VARCHAR); CREATE TABLE orders (id INT, customer_id INT, amount DECIMAL)",
        "question": "Liste clientes e seus pedidos",
    },
    {
        "cat": "GROUP BY",
        "schema": "CREATE TABLE sales (id INT, product VARCHAR, quantity INT, revenue DECIMAL)",
        "question": "Qual a receita total por produto?",
    },
    {
        "cat": "ORDER BY",
        "schema": "CREATE TABLE movies (id INT, title VARCHAR, rating DECIMAL, year INT)",
        "question": "Top 10 filmes mais bem avaliados",
    },
    {
        "cat": "SUBQUERY",
        "schema": "CREATE TABLE employees (id INT, name VARCHAR, salary DECIMAL, dept VARCHAR)",
        "question": "Funcionarios que ganham acima da media",
    },
    {
        "cat": "DISTINCT",
        "schema": "CREATE TABLE visits (id INT, user_id INT, page VARCHAR, date DATE)",
        "question": "Quantos usuarios unicos visitaram?",
    },
    {
        "cat": "HAVING",
        "schema": "CREATE TABLE transactions (id INT, account INT, amount DECIMAL, type VARCHAR)",
        "question": "Contas com total de transacoes acima de 10000",
    },
    {
        "cat": "LIKE",
        "schema": "CREATE TABLE books (id INT, title VARCHAR, author VARCHAR, isbn VARCHAR)",
        "question": "Livros cujo titulo comeca com 'SQL'",
    },
    {
        "cat": "IN",
        "schema": "CREATE TABLE students (id INT, name VARCHAR, grade VARCHAR, city VARCHAR)",
        "question": "Alunos das cidades Rio, Sao Paulo ou Brasilia",
    },
    {
        "cat": "BETWEEN",
        "schema": "CREATE TABLE events (id INT, name VARCHAR, date DATE, attendees INT)",
        "question": "Eventos entre 2023 e 2024",
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


def evaluate_sql(sql: str, category: str) -> Dict[str, bool]:
    sql_upper = sql.upper()
    has_select = "SELECT" in sql_upper
    has_from = "FROM" in sql_upper
    is_sql = has_select and has_from and "LET ME" not in sql_upper and "EXPLAIN" not in sql_upper
    has_keyword = category.upper() in sql_upper or category.lower() in sql.lower()
    return {"is_sql": is_sql, "has_keyword": has_keyword}


def format_sql(raw_sql: str) -> str:
    sql = raw_sql.replace("<think>", "").replace("</think>", "").strip()
    if ";" in sql:
        sql = sql.split(";", 1)[0] + ";"
    return sql.strip()


def load_model(checkpoint: Path | None):
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_PATH),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if checkpoint is not None:
        model = PeftModel.from_pretrained(model, str(checkpoint))
    return model


def iter_cases(tokenizer, test_cases: Iterable[Dict[str, str]]):
    for case in test_cases:
        messages = [
            {
                "role": "system",
                "content": f"Dialect: postgres\nSchema:\n{case['schema']}",
            },
            {"role": "user", "content": case["question"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        yield case, tokenizer([text], return_tensors="pt").to("cuda")


def main() -> None:
    print("=" * 100)
    print("Comparacao de checkpoints Expert-SQL")
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_PATH), trust_remote_code=True)

    results_summary = {}

    for label, ckpt_path in CHECKPOINTS.items():
        print(f"\n>> Avaliando '{label}'...")
        model = load_model(ckpt_path)
        model.eval()

        sql_hits = 0
        keyword_hits = 0
        detailed_rows = []

        with torch.inference_mode():
            for case, inputs in iter_cases(tokenizer, TEST_CASES):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=120,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                    pad_token_id=tokenizer.eos_token_id,
                )
                raw_sql = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                formatted = format_sql(raw_sql)
                eval_result = evaluate_sql(formatted, case["cat"])

                sql_hits += int(eval_result["is_sql"])
                keyword_hits += int(eval_result["has_keyword"])

                detailed_rows.append(
                    {
                        "cat": case["cat"],
                        "question": case["question"],
                        "sql": formatted,
                        **eval_result,
                    }
                )

        results_summary[label] = {
            "sql_hits": sql_hits,
            "keyword_hits": keyword_hits,
            "rows": detailed_rows,
        }

        del model
        torch.cuda.empty_cache()
        gc.collect()

    total_cases = len(TEST_CASES)
    print("\n" + "=" * 120)
    print(f"{'Checkpoint':<20} {'SQL Valido':<20} {'Keyword Correta':<25}")
    print("=" * 120)
    for label, data in results_summary.items():
        print(
            f"{label:<20} "
            f"{data['sql_hits']:>2}/{total_cases:<17} "
            f"{data['keyword_hits']:>2}/{total_cases:<22}"
        )
    print("=" * 120)

    # Detailed breakdown for best checkpoint
    best_label = max(results_summary.keys(), key=lambda k: results_summary[k]["sql_hits"])
    print(f"\nMelhor checkpoint: {best_label}")
    print("-" * 120)
    for row in results_summary[best_label]["rows"]:
        print(f"[{row['cat']}] SQL? {'SIM' if row['is_sql'] else 'NAO'} | KW? {'SIM' if row['has_keyword'] else 'NAO'}")
        print(f"Pergunta: {row['question']}")
        print(f"SQL: {row['sql']}\n")


if __name__ == "__main__":
    main()

