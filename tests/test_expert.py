"""Simple validation tests for SQL Expert"""

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

# Base model path (adjust as needed)
BASE_MODEL_PATH = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
EXPERT_PATH = "../weights/adapter"


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


def generate_sql(model, tokenizer, schema, question, max_tokens=200):
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
    # Extract only the generated part (after the prompt)
    sql = response.split("### SQL:")[-1].strip()
    
    return sql


class TestSQLExpert:
    """Test suite for SQL Expert"""
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        """Load model once for all tests"""
        return load_expert_model()
    
    def test_simple_select(self, expert_model):
        """Test basic SELECT query generation"""
        model, tokenizer = expert_model
        
        schema = "CREATE TABLE employees (id INTEGER, name VARCHAR, age INTEGER)"
        question = "Find all employees"
        sql = generate_sql(model, tokenizer, schema, question)
        
        assert "SELECT" in sql.upper()
        assert "employees" in sql.lower()
        assert "FROM" in sql.upper()
    
    def test_count_query(self, expert_model):
        """Test COUNT aggregation"""
        model, tokenizer = expert_model
        
        schema = "CREATE TABLE head (age INTEGER)"
        question = "How many heads of the departments are older than 56?"
        sql = generate_sql(model, tokenizer, schema, question)
        
        assert "SELECT" in sql.upper()
        assert "COUNT" in sql.upper()
        assert "FROM" in sql.upper()
        assert "head" in sql.lower()
        assert "WHERE" in sql.upper() or "age" in sql.lower()
    
    def test_filtering_query(self, expert_model):
        """Test query with WHERE clause"""
        model, tokenizer = expert_model
        
        schema = "CREATE TABLE employees (name VARCHAR, age INTEGER)"
        question = "Find employees older than 30"
        sql = generate_sql(model, tokenizer, schema, question)
        
        assert "SELECT" in sql.upper()
        assert "WHERE" in sql.upper()
        assert "employees" in sql.lower()
        assert "age" in sql.lower()
    
    def test_join_query(self, expert_model):
        """Test query with JOIN"""
        model, tokenizer = expert_model
        
        schema = """CREATE TABLE city (City_ID VARCHAR, Population INTEGER);
CREATE TABLE farm_competition (Theme VARCHAR, Host_city_ID VARCHAR)"""
        question = "Show themes of competitions in cities with population larger than 1000"
        sql = generate_sql(model, tokenizer, schema, question)
        
        assert "SELECT" in sql.upper()
        assert "JOIN" in sql.upper() or "FROM" in sql.upper()
        assert "WHERE" in sql.upper() or "1000" in sql
    
    def test_aggregation_with_group_by(self, expert_model):
        """Test query with GROUP BY"""
        model, tokenizer = expert_model
        
        schema = "CREATE TABLE employees (department VARCHAR, salary DECIMAL)"
        question = "What is the average salary by department?"
        sql = generate_sql(model, tokenizer, schema, question)
        
        assert "SELECT" in sql.upper()
        assert "AVG" in sql.upper() or "average" in sql.lower()
        assert "GROUP BY" in sql.upper() or "department" in sql.lower()
    
    def test_order_and_limit(self, expert_model):
        """Test query with ORDER BY and LIMIT"""
        model, tokenizer = expert_model
        
        schema = "CREATE TABLE products (name VARCHAR, price DECIMAL)"
        question = "Find top 5 most expensive products"
        sql = generate_sql(model, tokenizer, schema, question)
        
        assert "SELECT" in sql.upper()
        assert "products" in sql.lower()
        # Should have ordering and limit
        assert "ORDER" in sql.upper() or "LIMIT" in sql.upper() or "5" in sql
    
    def test_multiple_conditions(self, expert_model):
        """Test query with multiple WHERE conditions"""
        model, tokenizer = expert_model
        
        schema = "CREATE TABLE employees (name VARCHAR, age INTEGER, salary DECIMAL)"
        question = "Find employees older than 25 with salary above 50000"
        sql = generate_sql(model, tokenizer, schema, question)
        
        assert "SELECT" in sql.upper()
        assert "WHERE" in sql.upper()
        assert "AND" in sql.upper() or ("age" in sql.lower() and "salary" in sql.lower())
    
    def test_distinct_query(self, expert_model):
        """Test query with DISTINCT"""
        model, tokenizer = expert_model
        
        schema = "CREATE TABLE orders (customer_id INTEGER, product_id INTEGER)"
        question = "How many unique customers placed orders?"
        sql = generate_sql(model, tokenizer, schema, question)
        
        assert "SELECT" in sql.upper()
        assert "DISTINCT" in sql.upper() or "COUNT" in sql.upper()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


