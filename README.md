# Expert SQL

SQL query generation expert trained on validated multi-dialect SQL data (normalized PostgreSQL/MySQL/SQLite).

**Version:** 0.3.0 | **Checkpoint:** 500 | **Quality Score:** 9.6/10 | **Production Scenarios:** 30/30 passed

> ⚠️ Always deploy v0.3.0 (checkpoint-500). v0.2.1 (checkpoint-1250) regresses to text explanations.

## Quick Facts
- Base model `Qwen3-0.6B` with DoRA r=12 adapter (Unsloth, bf16)
- Grammar-constrained decoding using PostgreSQL-flavored GBNF
- Dataset: gretelai/synthetic_text_to_sql, Clinton/Text-to-sql-v1, synthetic_fixes
- Optimized for Windows + CUDA (RTX 4090 baseline), low VRAM footprint (~0.56 GB)
- Works best for e-commerce, CRM, analytics, operational reporting

## Version Comparison (CLI, English Prompts)

| Prompt | Base model (no expert) | v0.2.1 (`checkpoint-1250`) | v0.3.0 (`checkpoint-500`) |
|--------|-------------------------|----------------------------|---------------------------|
| `Schema: CREATE TABLE users (...); List all users.` | ❌ Narrative about app versions, no SQL | ❌ Lists fake user names, no SQL | ✅ `SELECT * FROM users;` |
| `Schema: CREATE TABLE products (...); Show products priced under 100.` | ❌ Describes how to round price values | ❌ Explains algorithm, no SQL | ✅ `SELECT * FROM products WHERE price < 100 AND stock > 0;` |
| `Schema: CREATE TABLE orders/customers (...); Show total revenue per customer.` | ❌ General explanation of totals | ❌ Repeats instructions, no SQL | ✅ `SELECT c.name, SUM(o.total) AS total_revenue FROM orders o JOIN customers c ON o.customer_id = c.id GROUP BY c.id;` |
| `Schema: CREATE TABLE customers/orders (...); Customers with more than 5 orders.` | ❌ Provides Markdown-style table text | ❌ Outputs Markdown-like table description | ⚠️ Generates SQL but may inject heuristic filters (e.g. `orders.total > 50000`) |

Run locally with:  
`expert-cli chat --experts sql@{version} --prompt "<prompt>" --max-tokens 120`

## Quick Start

```bash
# 1. Download package
wget https://github.com/hivellm/expert-sql/releases/download/v0.3.0/expert-sql-qwen3-0-6b.v0.3.0.expert

# 2. Install
expert-cli install expert-sql-qwen3-0-6b.v0.3.0.expert

# 3. Chat
expert-cli chat --experts sql
> List all users who registered in the last month
```

## Capabilities

- SELECT queries with WHERE, ORDER BY, LIMIT
- Robust INNER joins (2-4 tables) and multi-table aggregations
- Aggregations with GROUP BY, HAVING; COUNT, SUM, AVG, MIN, MAX
- Subqueries including EXISTS / NOT EXISTS, IN, correlated subqueries
- Date logic (EXTRACT, BETWEEN, INTERVAL) and string filtering (LIKE, CONCAT)
- Window functions (ROW_NUMBER, RANK, PARTITION BY)
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
- Recursive CTEs (`WITH RECURSIVE`) remain unreliable
- UNION / UNION ALL generates redundant predicates
- LEFT JOIN null-handling may degrade to INNER JOIN
- Deep (3+) CASE WHEN nesting still simplified
- Occasional ORDER BY alias mismatches—validate critical queries manually
- May inject heuristic numeric predicates (e.g. `orders.total > 50000`) on aggregate/count prompts

**Best practice:** always provide explicit schema context in prompts.

### Known Issues (v0.3.0, checkpoint-500)
- **Recursive CTEs:** still rewrites into self-joins or subqueries instead of proper recursion.
- **UNION / UNION ALL:** often adds redundant WHERE clauses or swaps to JOIN-based rewrites.
- **LEFT JOIN with NULL checks:** collapses back to INNER JOIN or misapplies IS NULL conditions.
- **NOT EXISTS:** occasionally combines INNER JOIN with NOT EXISTS, producing redundant logic.
- **Nested CASE WHEN:** consistent up to two levels; deeper nesting collapses to simpler branches.
- **Column aliases:** generally stable, but ORDER BY can reference outdated aliases in long queries.

These gaps were observed in manual QA and automated suites. Validate generated SQL when prompts require these structures; providing concrete column filters or expected patterns helps steer outputs.

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

`manifest.json` sets `packaging_checkpoint: "checkpoint-1250"` so packaging pulls the tuned checkpoint rather than the final training step.

### Package Layout

```
expert-sql-qwen3-0-6b.v0.3.0.expert
├── manifest.json
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── training_args.bin
├── vocab.json
├── README.md
├── grammar.gbnf
└── LICENSE
```

All artifacts live at the package root for loader compatibility (Linux, macOS, Windows).

### Integrity & Smoke Tests

```powershell
.\test_packaged_expert.ps1          # Extract, validate structure, run 3 inference checks
sha256sum -c expert-sql-qwen3-0-6b.v0.3.0.expert.sha256
```

## Testing & Benchmarking

```powershell
.\test.ps1 -TestSuite comparison       # Base vs expert regression suite
.\test.ps1 -TestSuite comprehensive    # 50+ SQL patterns
.\test.ps1 -TestSuite advanced         # Window functions, CTEs, set ops
.\run_all_tests.ps1 [-QuickTest|-SkipAdvanced]
.\benchmark_performance.ps1 [-Iterations 50] [-DetailedOutput]
```

Results summary (v0.3.0, checkpoint-500):
- 30/30 production scenarios succeed across e-commerce, CRM, analytics
- Syntax validity 100%, inference latency ~100-150 ms (RTX 4090)
- Complex scenario score average: 6.7/10 (strong on joins, weak on recursion/UNION)

## Troubleshooting
- **Quality dips:** confirm `use_unsloth: true`, avoid extending beyond 0.5 epoch without evaluation.
- **OOM on Windows:** batch size 2 + gradient accumulation 45; reduce sequence length or enable gradient checkpointing if needed.
- **Slow training:** verify CUDA availability (`nvidia-smi`), keep `torch_compile` disabled on Windows.
- **Console encoding:** use PowerShell and set UTF-8 (`[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`).

## Version History

### v0.3.0 (Production)
- Expanded dataset blend (gretelai, Clinton, synthetic_fixes)
- SQL dialect normalized to generic `sql`
- Checkpoint-500 selected (15/15 valid SQL, 9.6/10 quality)
- Improved deduplication (2,855 questions removed) and English-only prompts

### v0.2.0
- Packaging format flattened (no nested directories)
- Added manifest checkpoint selection
- 30/30 real-world scenarios validated with checkpoint-1250
- Known gaps: recursive CTEs, UNION, LEFT JOIN null handling

### v0.0.1
- Initial DoRA release leveraging checkpoint-500
- Baseline packaging layout with nested weights (deprecated)

## License

CC-BY-4.0
