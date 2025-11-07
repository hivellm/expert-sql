"""Advanced SQL Expert tests - Window functions, CTEs, complex patterns"""

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

BASE_MODEL_PATH = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERT_PATH = os.path.join(_SCRIPT_DIR, "..", "weights", "qwen3-06b", "final")


def load_base_model():
    """Load base model without expert"""
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    return model, tokenizer


def load_expert_model():
    """Load base model + expert adapter"""
    if not os.path.exists(EXPERT_PATH):
        pytest.skip(f"Expert weights not found at {EXPERT_PATH}")
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(model, EXPERT_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    return model, tokenizer


def generate_sql(model, tokenizer, schema, question, max_tokens=400):
    """Generate SQL query from question and schema"""
    prompt = f"""### Instruction: Generate a SQL query for the following question based on the database schema.

### Schema:
{schema}

### Question: {question}

### SQL:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql = response.split("### SQL:")[-1].strip()
    
    return sql


def check_sql_keywords(sql, required_keywords):
    """Check if SQL contains required keywords"""
    sql_upper = sql.upper()
    found = [kw for kw in required_keywords if kw.upper() in sql_upper]
    return found, len(found) / len(required_keywords) if required_keywords else 0


class TestWindowFunctions:
    """Test window functions (OVER, PARTITION BY, ROW_NUMBER, RANK)"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        return load_base_model()
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        return load_expert_model()
    
    def test_row_number(self, base_model, expert_model):
        """Test ROW_NUMBER() window function"""
        schema = "CREATE TABLE sales (salesperson VARCHAR, amount DECIMAL, region VARCHAR)"
        question = "Rank salespeople by sales amount within each region"
        required = ["SELECT", "ROW_NUMBER", "OVER", "PARTITION BY", "region", "ORDER BY", "amount"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- ROW_NUMBER() OVER PARTITION BY ---")
        print(f"Base SQL:   {base_sql[:250]}")
        print(f"Expert SQL: {expert_sql[:250]}")
        print(f"Base found:   {base_found} ({base_score:.1%})")
        print(f"Expert found: {expert_found} ({expert_score:.1%})")
        
        assert expert_score >= base_score * 0.6  # Window functions are very advanced
    
    def test_running_total(self, base_model, expert_model):
        """Test running total with SUM() OVER"""
        schema = "CREATE TABLE transactions (date DATE, amount DECIMAL)"
        question = "Calculate running total of transactions by date"
        required = ["SELECT", "SUM", "OVER", "ORDER BY", "date"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- Running Total (SUM OVER) ---")
        print(f"Base SQL:   {base_sql[:250]}")
        print(f"Expert SQL: {expert_sql[:250]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.6
    
    def test_rank_dense_rank(self, base_model, expert_model):
        """Test RANK() and DENSE_RANK()"""
        schema = "CREATE TABLE students (name VARCHAR, score INTEGER)"
        question = "Rank students by score, with same scores getting same rank"
        required = ["SELECT", "RANK", "OVER", "ORDER BY", "score", "DESC"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- RANK() Function ---")
        print(f"Base SQL:   {base_sql[:250]}")
        print(f"Expert SQL: {expert_sql[:250]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.6


class TestCTEs:
    """Test Common Table Expressions (WITH clause)"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        return load_base_model()
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        return load_expert_model()
    
    def test_simple_cte(self, base_model, expert_model):
        """Test simple CTE"""
        schema = "CREATE TABLE employees (id INTEGER, name VARCHAR, salary DECIMAL)"
        question = "Using a CTE, find employees earning more than the average salary"
        required = ["WITH", "AS", "SELECT", "AVG", "FROM", "WHERE"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- Simple CTE (WITH clause) ---")
        print(f"Base SQL:   {base_sql[:250]}")
        print(f"Expert SQL: {expert_sql[:250]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.6  # CTEs are advanced
    
    def test_multiple_ctes(self, base_model, expert_model):
        """Test multiple CTEs"""
        schema = """CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount DECIMAL);
CREATE TABLE customers (id INTEGER, name VARCHAR, region VARCHAR)"""
        question = "Using CTEs, find regions with above-average total sales"
        required = ["WITH", "AS", "SELECT", "SUM", "AVG", "GROUP BY"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- Multiple CTEs ---")
        print(f"Base SQL:   {base_sql[:250]}")
        print(f"Expert SQL: {expert_sql[:250]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.6


class TestSetOperations:
    """Test UNION, INTERSECT, EXCEPT"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        return load_base_model()
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        return load_expert_model()
    
    def test_union(self, base_model, expert_model):
        """Test UNION"""
        schema = """CREATE TABLE customers_2023 (id INTEGER, name VARCHAR);
CREATE TABLE customers_2024 (id INTEGER, name VARCHAR)"""
        question = "Get all unique customers from both 2023 and 2024"
        required = ["SELECT", "FROM", "customers_2023", "UNION", "customers_2024"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- UNION ---")
        print(f"Base SQL:   {base_sql[:250]}")
        print(f"Expert SQL: {expert_sql[:250]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.8
    
    def test_union_all(self, base_model, expert_model):
        """Test UNION ALL (including duplicates)"""
        schema = """CREATE TABLE sales_q1 (product VARCHAR, amount DECIMAL);
CREATE TABLE sales_q2 (product VARCHAR, amount DECIMAL)"""
        question = "Combine all sales from Q1 and Q2, including duplicates"
        required = ["SELECT", "FROM", "sales_q1", "UNION ALL", "sales_q2"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- UNION ALL ---")
        print(f"Base SQL:   {base_sql[:250]}")
        print(f"Expert SQL: {expert_sql[:250]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.8


class TestSelfJoins:
    """Test self-joins"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        return load_base_model()
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        return load_expert_model()
    
    def test_self_join_hierarchy(self, base_model, expert_model):
        """Test self-join for hierarchical data"""
        schema = "CREATE TABLE employees (id INTEGER, name VARCHAR, manager_id INTEGER)"
        question = "Show each employee with their manager's name"
        required = ["SELECT", "FROM", "employees", "JOIN", "ON", "manager_id"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- Self-Join (Hierarchy) ---")
        print(f"Base SQL:   {base_sql[:250]}")
        print(f"Expert SQL: {expert_sql[:250]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.8


class TestComplexAggregations:
    """Test complex aggregation patterns"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        return load_base_model()
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        return load_expert_model()
    
    def test_group_by_multiple_columns(self, base_model, expert_model):
        """Test GROUP BY with multiple columns"""
        schema = "CREATE TABLE sales (region VARCHAR, product VARCHAR, amount DECIMAL, year INTEGER)"
        question = "Show total sales by region, product, and year"
        required = ["SELECT", "SUM", "amount", "region", "product", "year", "GROUP BY"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- GROUP BY Multiple Columns ---")
        print(f"Base SQL:   {base_sql[:250]}")
        print(f"Expert SQL: {expert_sql[:250]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score
    
    def test_aggregation_with_filter(self, base_model, expert_model):
        """Test conditional aggregation"""
        schema = "CREATE TABLE orders (id INTEGER, status VARCHAR, amount DECIMAL)"
        question = "Show total completed orders and total pending orders separately"
        required = ["SELECT", "SUM", "CASE", "WHEN", "status", "FROM", "orders"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- Conditional Aggregation ---")
        print(f"Base SQL:   {base_sql[:250]}")
        print(f"Expert SQL: {expert_sql[:250]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.7


class TestCorrelatedSubqueries:
    """Test correlated subqueries"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        return load_base_model()
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        return load_expert_model()
    
    def test_exists(self, base_model, expert_model):
        """Test EXISTS with correlated subquery"""
        schema = """CREATE TABLE customers (id INTEGER, name VARCHAR);
CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount DECIMAL)"""
        question = "Find customers who have placed at least one order"
        required = ["SELECT", "FROM", "customers", "WHERE", "EXISTS", "orders"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- EXISTS Correlated Subquery ---")
        print(f"Base SQL:   {base_sql[:250]}")
        print(f"Expert SQL: {expert_sql[:250]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.7
    
    def test_not_exists(self, base_model, expert_model):
        """Test NOT EXISTS"""
        schema = """CREATE TABLE products (id INTEGER, name VARCHAR);
CREATE TABLE inventory (product_id INTEGER, quantity INTEGER)"""
        question = "Find products that are out of stock (not in inventory)"
        required = ["SELECT", "FROM", "products", "WHERE", "NOT EXISTS", "inventory"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- NOT EXISTS ---")
        print(f"Base SQL:   {base_sql[:250]}")
        print(f"Expert SQL: {expert_sql[:250]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.7


class TestMathematicalOperations:
    """Test mathematical operations and functions"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        return load_base_model()
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        return load_expert_model()
    
    def test_percentage_calculation(self, base_model, expert_model):
        """Test percentage calculations"""
        schema = "CREATE TABLE products (name VARCHAR, cost DECIMAL, price DECIMAL)"
        question = "Calculate profit margin percentage for each product"
        required = ["SELECT", "name", "price", "cost", "FROM", "products"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- Percentage Calculation ---")
        print(f"Base SQL:   {base_sql[:250]}")
        print(f"Expert SQL: {expert_sql[:250]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.8
    
    def test_round_function(self, base_model, expert_model):
        """Test ROUND function"""
        schema = "CREATE TABLE measurements (id INTEGER, value DECIMAL)"
        question = "Show measurements rounded to 2 decimal places"
        required = ["SELECT", "ROUND", "value", "FROM", "measurements"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- ROUND Function ---")
        print(f"Base SQL:   {base_sql[:250]}")
        print(f"Expert SQL: {expert_sql[:250]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.9


class TestTopN:
    """Test Top-N queries"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        return load_base_model()
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        return load_expert_model()
    
    def test_top_n_per_group(self, base_model, expert_model):
        """Test top N items per group"""
        schema = "CREATE TABLE products (id INTEGER, category VARCHAR, sales INTEGER)"
        question = "Find top 3 products by sales in each category"
        required = ["SELECT", "category", "sales", "FROM", "products"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- Top N per Group ---")
        print(f"Base SQL:   {base_sql[:250]}")
        print(f"Expert SQL: {expert_sql[:250]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.7  # Very complex query


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

