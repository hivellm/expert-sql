"""Comprehensive SQL Expert tests - Advanced SQL patterns"""

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


def generate_sql(model, tokenizer, schema, question, max_tokens=300):
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


class TestSubqueries:
    """Test subquery patterns"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        return load_base_model()
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        return load_expert_model()
    
    def test_subquery_in_where(self, base_model, expert_model):
        """Test subquery in WHERE clause"""
        schema = """CREATE TABLE employees (id INTEGER, name VARCHAR, salary DECIMAL, department_id INTEGER);
CREATE TABLE departments (id INTEGER, name VARCHAR, avg_salary DECIMAL)"""
        question = "Find employees who earn more than their department's average salary"
        required = ["SELECT", "FROM", "employees", "WHERE", "salary", ">", "SELECT", "AVG"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- Subquery in WHERE ---")
        print(f"Base SQL:   {base_sql[:200]}")
        print(f"Expert SQL: {expert_sql[:200]}")
        print(f"Base found:   {base_found} ({base_score:.1%})")
        print(f"Expert found: {expert_found} ({expert_score:.1%})")
        
        assert expert_score >= base_score * 0.8  # Allow some flexibility
    
    def test_subquery_in_from(self, base_model, expert_model):
        """Test subquery in FROM clause (derived table)"""
        schema = "CREATE TABLE sales (product_id INTEGER, amount DECIMAL, date DATE)"
        question = "What is the average of daily total sales?"
        required = ["SELECT", "AVG", "FROM", "SELECT", "SUM"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- Subquery in FROM (Derived Table) ---")
        print(f"Base SQL:   {base_sql[:200]}")
        print(f"Expert SQL: {expert_sql[:200]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.8


class TestMultipleJoins:
    """Test complex JOIN patterns"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        return load_base_model()
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        return load_expert_model()
    
    def test_three_table_join(self, base_model, expert_model):
        """Test joining three tables"""
        schema = """CREATE TABLE orders (id INTEGER, customer_id INTEGER, product_id INTEGER);
CREATE TABLE customers (id INTEGER, name VARCHAR);
CREATE TABLE products (id INTEGER, name VARCHAR, price DECIMAL)"""
        question = "List customer names, product names, and prices for all orders"
        required = ["SELECT", "customers", "products", "orders", "JOIN", "ON"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- Three Table JOIN ---")
        print(f"Base SQL:   {base_sql[:200]}")
        print(f"Expert SQL: {expert_sql[:200]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.9
    
    def test_left_join(self, base_model, expert_model):
        """Test LEFT JOIN (outer join)"""
        schema = """CREATE TABLE authors (id INTEGER, name VARCHAR);
CREATE TABLE books (id INTEGER, title VARCHAR, author_id INTEGER)"""
        question = "Show all authors and their books, including authors with no books"
        required = ["SELECT", "LEFT JOIN", "authors", "books"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- LEFT JOIN ---")
        print(f"Base SQL:   {base_sql[:200]}")
        print(f"Expert SQL: {expert_sql[:200]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.8


class TestAggregations:
    """Test various aggregation functions"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        return load_base_model()
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        return load_expert_model()
    
    def test_multiple_aggregations(self, base_model, expert_model):
        """Test multiple aggregation functions in one query"""
        schema = "CREATE TABLE products (category VARCHAR, price DECIMAL, stock INTEGER)"
        question = "For each category, show the average price, total stock, and number of products"
        required = ["SELECT", "AVG", "SUM", "COUNT", "category", "GROUP BY"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- Multiple Aggregations ---")
        print(f"Base SQL:   {base_sql[:200]}")
        print(f"Expert SQL: {expert_sql[:200]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score
    
    def test_having_clause(self, base_model, expert_model):
        """Test HAVING clause with GROUP BY"""
        schema = "CREATE TABLE sales (salesperson VARCHAR, amount DECIMAL)"
        question = "Which salespeople have total sales greater than 10000?"
        required = ["SELECT", "SUM", "GROUP BY", "HAVING", "10000"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- HAVING Clause ---")
        print(f"Base SQL:   {base_sql[:200]}")
        print(f"Expert SQL: {expert_sql[:200]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.9


class TestOrderingAndLimiting:
    """Test ORDER BY, LIMIT, OFFSET"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        return load_base_model()
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        return load_expert_model()
    
    def test_order_by_desc(self, base_model, expert_model):
        """Test ORDER BY DESC"""
        schema = "CREATE TABLE students (name VARCHAR, grade DECIMAL)"
        question = "Show top 10 students by grade, highest first"
        required = ["SELECT", "FROM", "students", "ORDER BY", "DESC", "LIMIT", "10"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- ORDER BY DESC + LIMIT ---")
        print(f"Base SQL:   {base_sql[:200]}")
        print(f"Expert SQL: {expert_sql[:200]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score
    
    def test_limit_offset(self, base_model, expert_model):
        """Test LIMIT with OFFSET (pagination)"""
        schema = "CREATE TABLE posts (id INTEGER, title VARCHAR, created_at DATE)"
        question = "Show posts 11-20 ordered by creation date"
        required = ["SELECT", "FROM", "posts", "ORDER BY", "LIMIT", "OFFSET"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- LIMIT OFFSET (Pagination) ---")
        print(f"Base SQL:   {base_sql[:200]}")
        print(f"Expert SQL: {expert_sql[:200]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.8


class TestDistinctAndUnion:
    """Test DISTINCT and UNION operations"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        return load_base_model()
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        return load_expert_model()
    
    def test_distinct(self, base_model, expert_model):
        """Test SELECT DISTINCT"""
        schema = "CREATE TABLE orders (customer_id INTEGER, product_id INTEGER)"
        question = "How many unique customers have placed orders?"
        required = ["SELECT", "COUNT", "DISTINCT", "customer_id", "FROM", "orders"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- COUNT DISTINCT ---")
        print(f"Base SQL:   {base_sql[:200]}")
        print(f"Expert SQL: {expert_sql[:200]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score


class TestCaseStatements:
    """Test CASE WHEN statements"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        return load_base_model()
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        return load_expert_model()
    
    def test_case_categorization(self, base_model, expert_model):
        """Test CASE for categorization"""
        schema = "CREATE TABLE products (name VARCHAR, price DECIMAL)"
        question = "Categorize products as 'cheap' (under 10), 'medium' (10-50), or 'expensive' (over 50)"
        required = ["SELECT", "CASE", "WHEN", "price", "THEN", "END", "FROM", "products"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- CASE WHEN Categorization ---")
        print(f"Base SQL:   {base_sql[:200]}")
        print(f"Expert SQL: {expert_sql[:200]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.7  # CASE is complex


class TestNullHandling:
    """Test NULL handling"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        return load_base_model()
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        return load_expert_model()
    
    def test_is_null(self, base_model, expert_model):
        """Test IS NULL check"""
        schema = "CREATE TABLE users (id INTEGER, name VARCHAR, email VARCHAR)"
        question = "Find all users without an email address"
        required = ["SELECT", "FROM", "users", "WHERE", "email", "IS NULL"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- IS NULL Check ---")
        print(f"Base SQL:   {base_sql[:200]}")
        print(f"Expert SQL: {expert_sql[:200]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.9
    
    def test_coalesce(self, base_model, expert_model):
        """Test COALESCE for NULL handling"""
        schema = "CREATE TABLE products (name VARCHAR, discount DECIMAL, regular_price DECIMAL)"
        question = "Show product names and their effective price (discount price if available, otherwise regular price)"
        required = ["SELECT", "COALESCE", "discount", "regular_price", "FROM", "products"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- COALESCE ---")
        print(f"Base SQL:   {base_sql[:200]}")
        print(f"Expert SQL: {expert_sql[:200]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.7  # COALESCE is advanced


class TestDateFunctions:
    """Test date/time functions"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        return load_base_model()
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        return load_expert_model()
    
    def test_date_filtering(self, base_model, expert_model):
        """Test date range filtering"""
        schema = "CREATE TABLE events (id INTEGER, name VARCHAR, event_date DATE)"
        question = "Find events in the year 2024"
        required = ["SELECT", "FROM", "events", "WHERE", "event_date", "2024"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- Date Filtering ---")
        print(f"Base SQL:   {base_sql[:200]}")
        print(f"Expert SQL: {expert_sql[:200]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.9


class TestStringFunctions:
    """Test string manipulation functions"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        return load_base_model()
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        return load_expert_model()
    
    def test_like_pattern(self, base_model, expert_model):
        """Test LIKE pattern matching"""
        schema = "CREATE TABLE users (id INTEGER, email VARCHAR, name VARCHAR)"
        question = "Find all users with Gmail addresses"
        required = ["SELECT", "FROM", "users", "WHERE", "email", "LIKE", "@gmail"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- LIKE Pattern Matching ---")
        print(f"Base SQL:   {base_sql[:200]}")
        print(f"Expert SQL: {expert_sql[:200]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.9
    
    def test_concatenation(self, base_model, expert_model):
        """Test string concatenation"""
        schema = "CREATE TABLE users (first_name VARCHAR, last_name VARCHAR)"
        question = "Show full names by combining first and last names"
        required = ["SELECT", "first_name", "last_name", "FROM", "users"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- String Concatenation ---")
        print(f"Base SQL:   {base_sql[:200]}")
        print(f"Expert SQL: {expert_sql[:200]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.8


class TestComplexFiltering:
    """Test complex WHERE conditions"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        return load_base_model()
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        return load_expert_model()
    
    def test_in_clause(self, base_model, expert_model):
        """Test IN clause"""
        schema = "CREATE TABLE products (id INTEGER, category VARCHAR, price DECIMAL)"
        question = "Find products in categories electronics, books, or toys"
        required = ["SELECT", "FROM", "products", "WHERE", "category", "IN"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- IN Clause ---")
        print(f"Base SQL:   {base_sql[:200]}")
        print(f"Expert SQL: {expert_sql[:200]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score
    
    def test_and_or_conditions(self, base_model, expert_model):
        """Test AND/OR combinations"""
        schema = "CREATE TABLE employees (name VARCHAR, department VARCHAR, salary DECIMAL, years INTEGER)"
        question = "Find employees in sales or marketing department with salary > 50000 and more than 2 years experience"
        required = ["SELECT", "FROM", "employees", "WHERE", "AND", "OR", "salary", "50000", "years", "2"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_found, base_score = check_sql_keywords(base_sql, required)
        expert_found, expert_score = check_sql_keywords(expert_sql, required)
        
        print(f"\n--- AND/OR Combinations ---")
        print(f"Base SQL:   {base_sql[:200]}")
        print(f"Expert SQL: {expert_sql[:200]}")
        print(f"Base score:   {base_score:.1%}")
        print(f"Expert score: {expert_score:.1%}")
        
        assert expert_score >= base_score * 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

