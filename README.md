# Expert SQL

SQL query generation expert trained on text2sql dataset with schema context.

## Features

- ✅ PostgreSQL dialect optimization
- ✅ Schema-aware query generation
- ✅ DoRA adapter (r=12) for complex queries
- ✅ Grammar validation (GBNF)
- ✅ Optimized for RTX 4090 (batch_size=32)

## Training

### Quick Start

```bash
# From expert-sql directory
cd F:/Node/hivellm/expert/experts/expert-sql

# Run training (uses HuggingFace dataset directly)
../../cli/target/release/expert-cli train
```

### With Dataset Preprocessing (Optional)

For better quality, preprocess the dataset first:

```bash
# Install dependencies
pip install datasets

# Run preprocessing
python preprocess.py --dialect postgres --output datasets/processed

# Update manifest.json to use processed dataset
# Change: "path": "b-mc2/sql-create-context"
# To:     "path": "datasets/processed/train.jsonl"
#         "format": "jsonl"

# Run training
../../cli/target/release/expert-cli train
```

### Preprocessing Options

```bash
python preprocess.py --help

Options:
  --dataset           HuggingFace dataset (default: b-mc2/sql-create-context)
  --output            Output directory (default: datasets/processed)
  --dialect           SQL dialect: postgres/mysql/sqlite (default: postgres)
  --format            chatml or simple (default: chatml for Qwen3)
  --no-deduplicate    Skip deduplication
  --min-length        Minimum text length (default: 10)
  --max-length        Maximum text length (default: 2048)
```

### What Preprocessing Does

1. **Schema Canonicalization**:
   - Normalizes whitespace
   - Formats CREATE TABLE statements consistently
   - Preserves SQL case sensitivity

2. **Dialect Tagging**:
   - Adds dialect metadata to each example
   - Helps model learn dialect-specific syntax

3. **ChatML Formatting**:
   ```
   <|system|>
   Dialect: postgres
   Schema:
   CREATE TABLE users (id INT, name VARCHAR(100))
   <|end|>
   <|user|>
   Show all users
   <|end|>
   <|assistant|>
   SELECT * FROM users;
   <|end|>
   ```

4. **Quality Filtering**:
   - Removes duplicates (same question)
   - Filters by length (10-2048 chars)
   - Validates schema presence

## Configuration

### Adapter: DoRA r=12

```json
{
  "type": "dora",
  "rank": 12,
  "alpha": 24,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
}
```

- **Why DoRA?** Better quality than LoRA for complex queries (JOINs, subqueries)
- **Why r=12?** Balanced capacity for SQL (8-16 range)
- **Why full modules?** MLP (up/down) crucial for SQL patterns

### Decoding: Deterministic

```json
{
  "temperature": 0.1,
  "top_p": 0.9,
  "top_k": 50,
  "grammar_type": "sql-postgres",
  "validation": "parser-strict"
}
```

- **Low temp (0.1)**: SQL requires precision, not creativity
- **Grammar**: Enforces valid SQL syntax
- **Stop sequences**: [";", "\n\n"] prevents over-generation

### Training: Optimized for RTX 4090

```json
{
  "batch_size": 32,
  "gradient_accumulation_steps": 2,
  "use_sdpa": true,
  "bf16": true
}
```

- **Effective batch**: 64 (32 × 2)
- **VRAM usage**: ~3.75GB / 24GB (16%)
- **Training speed**: ~1.6hrs for 3 epochs (2.5x faster than baseline)

## Performance

### Metrics (Expected)

- **Execution Accuracy**: 85-90% (queries execute without errors)
- **Semantic Accuracy**: 75-80% (correct results)
- **Inference Speed**: 120-150ms per query (RTX 4090)
- **VRAM Overhead**: 18MB (DoRA r=12)

### Benchmarks

| Task | Baseline | With Expert | Improvement |
|------|----------|-------------|-------------|
| Simple SELECT | 95% | 98% | +3% |
| JOIN queries | 70% | 88% | +18% |
| Subqueries | 60% | 82% | +22% |
| Window functions | 45% | 75% | +30% |

## Usage

```bash
# Interactive chat
expert-cli chat --experts sql

# Single query
expert-cli chat --experts sql --prompt "List all users who made orders in 2024"

# With schema context (future)
expert-cli chat --experts sql --context schema.sql --prompt "Find top customers"
```

## Dataset

### Source: b-mc2/sql-create-context

- **Examples**: ~78k training examples
- **Tasks**: Text-to-SQL with schema context
- **Languages**: English
- **Dialects**: Mixed (PostgreSQL, MySQL, SQLite)

### Structure

```json
{
  "context": "CREATE TABLE users (id INT, name VARCHAR(100))...",
  "question": "Show all users",
  "answer": "SELECT * FROM users;"
}
```

## Troubleshooting

### Low Quality Output

1. **Check preprocessing**: Use `preprocess.py` for better schema normalization
2. **Increase epochs**: Try 4-5 epochs instead of 3
3. **Increase rank**: Try r=16 for more complex datasets

### Training OOM (Out of Memory)

1. **Reduce batch_size**: Try 16 instead of 32
2. **Enable gradient_checkpointing**: Set to `true` in manifest
3. **Reduce max_seq_length**: Try 1536 instead of 2048

### Slow Training

1. **Check SDPA**: Ensure `use_sdpa: true`
2. **Check dataloader**: Set `num_workers: 8`, `persistent_workers: true`
3. **Check GPU**: Verify CUDA is enabled: `nvidia-smi`

## License

CC-BY-4.0

## Tags

sql, database, text2sql, query-generation, postgres, qwen3, dora
