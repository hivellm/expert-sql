# Expert SQL

[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](https://github.com/hivellm/expert-sql/releases/tag/v0.3.0)
[![License](https://img.shields.io/badge/license-CC--BY--4.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production-brightgreen.svg)](README.md#version-history)
[![Quality Score](https://img.shields.io/badge/quality-8.9%2F10-brightgreen.svg)](README.md#testing--benchmarking)
[![Test Scenarios](https://img.shields.io/badge/tests-16%2F18%20passed-success.svg)](README.md#testing--benchmarking)

[![Base Model](https://img.shields.io/badge/base%20model-Qwen3--0.6B-orange.svg)](README.md#quick-facts)
[![Checkpoint](https://img.shields.io/badge/checkpoint-1000-blue.svg)](README.md#version-history)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20CUDA-0078d4.svg)](README.md#quick-facts)
[![VRAM](https://img.shields.io/badge/VRAM-~0.56%20GB-yellow.svg)](README.md#quick-facts)

SQL query generation expert trained on validated multi-dialect SQL data (normalized PostgreSQL/MySQL/SQLite).

> ✅ Always deploy v0.3.0 (checkpoint-1000). Best overall quality and consistency compared to checkpoints 750, 1250, and 1496.

## Quick Facts
- Base model `Qwen3-0.6B` with DoRA r=12 adapter (Unsloth, bf16)
- Grammar-constrained decoding using PostgreSQL-flavored GBNF
- Dataset: gretelai/synthetic_text_to_sql, Clinton/Text-to-sql-v1, synthetic_fixes
- Optimized for Windows + CUDA (RTX 4090 baseline), low VRAM footprint (~0.56 GB)
- Works best for e-commerce, CRM, analytics, operational reporting

## Quick Start

```bash
# Download and install
wget https://github.com/hivellm/expert-sql/releases/download/v0.3.0/expert-sql-qwen3-0-6b.v0.3.0.expert
expert-cli install expert-sql-qwen3-0-6b.v0.3.0.expert

# Use
expert-cli chat --experts sql
```

## Capabilities

- SELECT queries with WHERE, ORDER BY, LIMIT
- Robust INNER joins (2-4 tables) and multi-table aggregations
- Aggregations with GROUP BY, HAVING; COUNT, SUM, AVG, MIN, MAX
- Subqueries including EXISTS / NOT EXISTS, IN, correlated subqueries
- Date logic (EXTRACT, BETWEEN, INTERVAL) and string filtering (LIKE, CONCAT)
- LEFT JOIN with NULL checks
- Non-recursive CTEs and business-style reporting prompts

### Sample Output

```sql
-- Customers without orders
SELECT c.name
FROM customers c
WHERE NOT EXISTS (
  SELECT 1
  FROM orders o
  WHERE o.customer_id = c.id
);

-- Top revenue customers
SELECT c.name, SUM(o.total) AS total_spent
FROM orders o
JOIN customers c ON o.customer_id = c.id
GROUP BY c.name
ORDER BY total_spent DESC
LIMIT 10;
```

## Limitations

**Confirmed failures (tested via `test_limitations_cli.ps1`):**

- ❌ **Recursive CTEs (`WITH RECURSIVE`)** - Generates explanatory text instead of SQL
- ❌ **UNION / UNION ALL** - Generates explanatory text instead of SQL
- ❌ **Deep CASE WHEN nesting (3+ levels)** - Generates multiple-choice responses instead of SQL
- ❌ **Heuristic numeric predicates** - Generates only numbers instead of SQL queries (e.g., "more than 5 orders")
- ❌ **Window functions** - Generates explanatory text or uses GROUP BY incorrectly

**Working features:**

- ✅ **LEFT JOIN with NULL checks** - Correctly generates SQL with LEFT JOIN and WHERE IS NULL

**Best practice:** Always provide explicit schema context in prompts. For complex queries requiring the above features, consider alternative approaches or manual query construction.

## Training & Configuration

### Training Quick Start

```bash
cd F:/Node/hivellm/expert/experts/expert-sql
../../cli/target/release/expert-cli train
```

### Dataset
- Multi-source corpus stored at `datasets/train.jsonl`
- Preprocessed with sqlglot (dialect normalization, validation)
- Deduplicated by question (2,855 removed) and formatted in ChatML
- English-only prompts (Portuguese filtered out)

### Adapter Configuration (`adapter_config.json`)

```json
{
  "type": "dora",
  "rank": 12,
  "alpha": 24,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
}
```

### Recommended Decoding

```json
{
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "grammar_type": "sql-postgres",
  "validation": "parser-strict",
  "stop_sequences": [";", "\n\n"]
}
```

### Training Hyperparameters

```json
{
  "use_unsloth": true,
  "batch_size": 2,
  "gradient_accumulation_steps": 45,
  "learning_rate": 5e-5,
  "warmup_ratio": 0.1,
  "dropout": 0.1,
  "epochs": 1.5,
  "lr_scheduler": "cosine",
  "use_sdpa": true,
  "bf16": true,
  "torch_compile": false
}
```

Effective batch size: 90. Training completes in ~2-3 hours on RTX 4090 with Unsloth; peak VRAM ~0.56 GB.

## Packaging & Distribution

### Build Package

```bash
cd F:/Node/hivellm/expert/experts/expert-sql
../../cli/target/release/expert-cli package --manifest manifest.json --weights weights
# Outputs expert-sql-qwen3-0-6b.v0.3.0.expert (~26 MB) + .sha256
```

Packaging uses checkpoint-1000 (best quality from comparative analysis).

Package includes: manifest.json, adapter_model.safetensors, tokenizer files, grammar.gbnf, and LICENSE. All artifacts at package root for cross-platform compatibility.

### Integrity Tests

```powershell
.\test_packaged_expert.ps1
sha256sum -c expert-sql-qwen3-0-6b.v0.3.0.expert.sha256
```

## Testing & Benchmarking

```powershell
.\test.ps1 -TestSuite comparison       # Base vs expert regression suite
.\test.ps1 -TestSuite comprehensive    # 50+ SQL patterns
.\test.ps1 -TestSuite advanced         # Window functions, CTEs, set ops
.\test_limitations_cli.ps1             # Test confirmed limitations (Recursive CTEs, UNION, Window functions, etc.)
.\run_all_tests.ps1 [-QuickTest|-SkipAdvanced]
.\benchmark_performance.ps1 [-Iterations 50] [-DetailedOutput]
```

**Results (v0.3.0, checkpoint-1000):**
- 16/18 test scenarios correct (88.9% success rate)
- Best quality compared to checkpoints 750, 1250, 1496
- Inference latency ~100-150 ms (RTX 4090)
- Strong on basic SELECT, WHERE, JOIN, GROUP BY, subqueries

## Troubleshooting
- **Quality dips:** confirm `use_unsloth: true`, avoid extending beyond 0.5 epoch without evaluation.
- **OOM on Windows:** batch size 2 + gradient accumulation 45; reduce sequence length or enable gradient checkpointing if needed.
- **Slow training:** verify CUDA availability (`nvidia-smi`), keep `torch_compile` disabled on Windows.
- **Console encoding:** use PowerShell and set UTF-8 (`[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`).

## Version History

### v0.3.0 (Production)
- Expanded dataset blend (gretelai, Clinton, synthetic_fixes)
- SQL dialect normalized to PostgreSQL
- Checkpoint-1000 selected (16/18 correct, 88.9% success rate)
- Best quality compared to checkpoints 750, 1250, 1496
- Improved deduplication (2,855 questions removed)

### v0.2.0
- Packaging format flattened (no nested directories)
- Added manifest checkpoint selection
- Known gaps: recursive CTEs, UNION, LEFT JOIN null handling

### v0.0.1
- Initial DoRA release
- Baseline packaging layout with nested weights (deprecated)

## License

CC-BY-4.0
