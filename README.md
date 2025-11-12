# Expert SQL

SQL query generation expert trained on validated multi-dialect SQL data (normalized PostgreSQL/MySQL/SQLite).

**Version:** 0.3.0 | **Checkpoint:** 1000 | **Quality Score:** 8.9/10 | **Test Scenarios:** 16/18 passed (88.9%)

> ✅ Always deploy v0.3.0 (checkpoint-1000). Best overall quality and consistency compared to checkpoints 750, 1250, and 1496.

## Quick Facts
- Base model `Qwen3-0.6B` with DoRA r=12 adapter (Unsloth, bf16)
- Grammar-constrained decoding using PostgreSQL-flavored GBNF
- Dataset: gretelai/synthetic_text_to_sql, Clinton/Text-to-sql-v1, synthetic_fixes
- Optimized for Windows + CUDA (RTX 4090 baseline), low VRAM footprint (~0.56 GB)
- Works best for e-commerce, CRM, analytics, operational reporting

## Version Comparison (CLI, English Prompts)

| Prompt | Base model (no expert) | v0.3.0 (`checkpoint-1000`) |
|--------|-------------------------|---------------------------|
| `Schema: CREATE TABLE users (...); List all users.` | ❌ Narrative about app versions, no SQL | ✅ `SELECT * FROM users;` |
| `Schema: CREATE TABLE products (...); Show products priced under 100.` | ❌ Describes how to round price values | ✅ `SELECT * FROM products WHERE price < 100;` |
| `Schema: CREATE TABLE orders/customers (...); Show total revenue per customer.` | ❌ General explanation of totals | ✅ `SELECT c.name, SUM(o.total) AS total_revenue FROM orders o JOIN customers c ON o.customer_id = c.id GROUP BY c.id;` |
| `Schema: CREATE TABLE customers/orders (...); Customers with more than 5 orders.` | ❌ Provides Markdown-style table text | ⚠️ Generates SQL but may inject heuristic filters (e.g. `orders.total > 50000`) |

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
- LEFT JOIN with NULL checks (tested and confirmed working)
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

**Tested and Confirmed (v0.3.0, checkpoint-1000):**

- **Recursive CTEs (`WITH RECURSIVE`):** ❌ **FAILS** - Generates explanatory text instead of SQL with recursive CTEs. Cannot handle hierarchical queries requiring recursive traversal.
- **UNION / UNION ALL:** ❌ **FAILS** - Generates explanatory text instead of SQL with UNION operations. Cannot combine results from multiple tables using set operations.
- **Deep CASE WHEN nesting (3+ levels):** ❌ **FAILS** - Generates multiple-choice responses instead of SQL with nested CASE WHEN statements. Limited to simple conditional logic.
- **Heuristic numeric predicates:** ❌ **FAILS** - When asked for queries with numeric aggregations (e.g., "more than 5 orders"), generates only numbers instead of SQL queries. Cannot properly translate heuristic requirements into SQL.
- **Window functions (ROW_NUMBER() OVER, etc.):** ❌ **FAILS** - Does not generate SQL with window functions. Instead generates explanatory text or uses GROUP BY incorrectly.

**Working Features (Tested and Confirmed):**

- **LEFT JOIN with NULL checks:** ✅ **WORKS** - Correctly generates SQL with LEFT JOIN and WHERE IS NULL clauses to find records without matching relationships.

**Not Fully Tested:**

- **ORDER BY alias handling:** Basic ORDER BY works correctly, but alias handling not fully tested in complex scenarios.

**Best practice:** Always provide explicit schema context in prompts. For complex queries requiring recursive CTEs, UNION operations, window functions, or deep CASE WHEN nesting, consider alternative approaches or manual query construction.

### Known Issues (v0.3.0, checkpoint-1000)

Test results from `test_limitations_cli.ps1`:

- **Recursive CTEs:** ❌ **CONFIRMED** - Does not generate SQL with `WITH RECURSIVE`. Generates explanatory text instead.
- **UNION / UNION ALL:** ❌ **CONFIRMED** - Does not generate SQL with UNION operations. Generates explanatory text instead.
- **LEFT JOIN with NULL checks:** ✅ **CONFIRMED WORKING** - Correctly generates SQL with LEFT JOIN and WHERE IS NULL.
- **Nested CASE WHEN (3+ levels):** ❌ **CONFIRMED** - Does not generate SQL with nested CASE WHEN. Generates multiple-choice responses instead.
- **Heuristic numeric predicates:** ❌ **CONFIRMED** - When prompts require numeric aggregations with conditions (e.g., "more than 5 orders"), generates only numbers instead of SQL queries.
- **Window functions:** ❌ **CONFIRMED** - Does not generate SQL with ROW_NUMBER() OVER or other window functions. Generates explanatory text instead.
- **ORDER BY:** Partially tested - Basic ORDER BY works correctly, but alias handling not fully tested.

These limitations were confirmed through direct testing with `expert-cli chat` using limitation test cases. When prompts require these structures, the expert may generate explanatory text or incorrect output instead of valid SQL. Validate generated SQL and consider alternative query approaches for complex requirements.

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

`manifest.json` sets `packaging_checkpoint: "checkpoint-1000"` so packaging pulls the best quality checkpoint selected after comparative analysis.

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
.\test_limitations_cli.ps1             # Test confirmed limitations (Recursive CTEs, UNION, Window functions, etc.)
.\run_all_tests.ps1 [-QuickTest|-SkipAdvanced]
.\benchmark_performance.ps1 [-Iterations 50] [-DetailedOutput]
```

Results summary (v0.3.0, checkpoint-1000):
- 16/18 test scenarios correct (88.9% success rate) across basic and intermediate queries
- Best overall quality compared to checkpoints 750, 1250, 1496
- Syntax validity high, inference latency ~100-150 ms (RTX 4090)
- Strong on basic SELECT, WHERE, JOIN, GROUP BY, subqueries, LEFT JOIN with NULL checks
- **Limitations confirmed:** Recursive CTEs, UNION operations, window functions, deep CASE WHEN nesting, and heuristic numeric predicates do not work correctly

**Limitation test results:** See `test_limitations_cli.ps1` for detailed test cases. Most complex SQL features generate explanatory text instead of SQL.

## Troubleshooting
- **Quality dips:** confirm `use_unsloth: true`, avoid extending beyond 0.5 epoch without evaluation.
- **OOM on Windows:** batch size 2 + gradient accumulation 45; reduce sequence length or enable gradient checkpointing if needed.
- **Slow training:** verify CUDA availability (`nvidia-smi`), keep `torch_compile` disabled on Windows.
- **Console encoding:** use PowerShell and set UTF-8 (`[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`).

## Version History

### v0.3.0 (Production)
- Expanded dataset blend (gretelai, Clinton, synthetic_fixes)
- SQL dialect normalized to PostgreSQL
- Checkpoint-1000 selected after comparative analysis (16/18 correct, 88.9% success rate)
- Best overall quality and consistency compared to checkpoints 750, 1250, 1496
- Improved deduplication (2,855 questions removed) and English-only prompts
- **Limitation tests confirmed:** Recursive CTEs, UNION operations, window functions, deep CASE WHEN nesting, and heuristic numeric predicates do not work correctly (tested via `test_limitations_cli.ps1`)
- **Working features confirmed:** LEFT JOIN with NULL checks works correctly

### v0.2.0
- Packaging format flattened (no nested directories)
- Added manifest checkpoint selection
- Known gaps: recursive CTEs, UNION, LEFT JOIN null handling

### v0.0.1
- Initial DoRA release
- Baseline packaging layout with nested weights (deprecated)

## License

CC-BY-4.0
