"""Comprehensive test suite for SQL Expert - Hard queries"""

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

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


class TestSQLHardQueries:
    """Hard test cases for SQL Expert"""
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        """Load model once for all tests"""
        return load_expert_model()
    
    def test_multi_table_join(self, expert_model):
        """Test complex multi-table JOIN"""
        model, tokenizer = expert_model
        
        schema = """CREATE TABLE department (department_id VARCHAR, name VARCHAR, budget_in_billions VARCHAR);
CREATE TABLE management (department_id VARCHAR, head_id VARCHAR, temporary_acting VARCHAR);
CREATE TABLE head (head_id VARCHAR, name VARCHAR, born_state VARCHAR, age VARCHAR)"""
        
        question = "What are the distinct creation years of departments managed by a secretary born in Alabama?"
        sql = generate_sql(model, tokenizer, schema, question)
        
        print(f"\nGenerated SQL: {sql}")
        
        assert "SELECT" in sql.upper()
        assert "JOIN" in sql.upper()
        assert "Alabama" in sql or "born_state" in sql.lower()
    
    def test_subquery(self, expert_model):
        """Test query with subquery"""
        model, tokenizer = expert_model
        
        schema = """CREATE TABLE management (department_id VARCHAR);
CREATE TABLE department (department_id VARCHAR)"""
        
        question = "How many departments are led by heads who are not mentioned?"
        sql = generate_sql(model, tokenizer, schema, question)
        
        print(f"\nGenerated SQL: {sql}")
        
        assert "SELECT" in sql.upper()
        assert ("NOT IN" in sql.upper() or "NOT EXISTS" in sql.upper() or 
                "EXCEPT" in sql.upper() or "WHERE NOT" in sql.upper())
    
    def test_having_clause(self, expert_model):
        """Test query with HAVING clause"""
        model, tokenizer = expert_model
        
        schema = "CREATE TABLE head (born_state VARCHAR)"
        question = "What are the names of states where at least 3 heads were born?"
        sql = generate_sql(model, tokenizer, schema, question)
        
        print(f"\nGenerated SQL: {sql}")
        
        assert "SELECT" in sql.upper()
        assert "GROUP BY" in sql.upper()
        assert ("HAVING" in sql.upper() or "COUNT" in sql.upper())
    
    def test_intersect_query(self, expert_model):
        """Test query with INTERSECT"""
        model, tokenizer = expert_model
        
        schema = """CREATE TABLE department (department_id VARCHAR, name VARCHAR);
CREATE TABLE management (department_id VARCHAR, head_id VARCHAR);
CREATE TABLE head (head_id VARCHAR, born_state VARCHAR)"""
        
        question = "List states where both Treasury and Homeland Security secretaries were born"
        sql = generate_sql(model, tokenizer, schema, question)
        
        print(f"\nGenerated SQL: {sql}")
        
        assert "SELECT" in sql.upper()
        # May use INTERSECT or other approach
        assert ("INTERSECT" in sql.upper() or 
                ("Treasury" in sql and "Homeland" in sql))
    
    def test_complex_aggregation(self, expert_model):
        """Test complex aggregation with multiple functions"""
        model, tokenizer = expert_model
        
        schema = "CREATE TABLE department (budget_in_billions INTEGER)"
        question = "What are the maximum and minimum budget of departments?"
        sql = generate_sql(model, tokenizer, schema, question)
        
        print(f"\nGenerated SQL: {sql}")
        
        assert "SELECT" in sql.upper()
        assert "MAX" in sql.upper()
        assert "MIN" in sql.upper()
        assert "budget" in sql.lower()
    
    def test_like_pattern_matching(self, expert_model):
        """Test LIKE pattern matching"""
        model, tokenizer = expert_model
        
        schema = "CREATE TABLE head (head_id VARCHAR, name VARCHAR)"
        question = "Which head's name contains 'Ha'? List the id and name"
        sql = generate_sql(model, tokenizer, schema, question)
        
        print(f"\nGenerated SQL: {sql}")
        
        assert "SELECT" in sql.upper()
        assert ("LIKE" in sql.upper() or "'%Ha%'" in sql or '"%Ha%"' in sql)
    
    def test_self_join(self, expert_model):
        """Test self-join query"""
        model, tokenizer = expert_model
        
        schema = """CREATE TABLE flight (flno INTEGER, origin VARCHAR, destination VARCHAR);
CREATE TABLE aircraft (aid INTEGER, name VARCHAR, distance INTEGER)"""
        
        question = "Find the name of aircraft that can cover the furthest distance"
        sql = generate_sql(model, tokenizer, schema, question)
        
        print(f"\nGenerated SQL: {sql}")
        
        assert "SELECT" in sql.upper()
        assert "aircraft" in sql.lower()
        assert ("MAX" in sql.upper() or "ORDER BY" in sql.upper() or "LIMIT" in sql.upper())
    
    def test_count_distinct(self, expert_model):
        """Test COUNT DISTINCT"""
        model, tokenizer = expert_model
        
        schema = "CREATE TABLE management (temporary_acting VARCHAR)"
        question = "How many different acting statuses are there?"
        sql = generate_sql(model, tokenizer, schema, question)
        
        print(f"\nGenerated SQL: {sql}")
        
        assert "SELECT" in sql.upper()
        assert "COUNT" in sql.upper()
        assert "DISTINCT" in sql.upper()
    
    def test_multiple_joins_with_aggregation(self, expert_model):
        """Test multiple JOINs with aggregation"""
        model, tokenizer = expert_model
        
        schema = """CREATE TABLE management (department_id VARCHAR, head_id VARCHAR);
CREATE TABLE department (department_id VARCHAR, name VARCHAR);
CREATE TABLE head (head_id VARCHAR, age VARCHAR)"""
        
        question = "Find departments with more than 1 head at a time and show department name and count"
        sql = generate_sql(model, tokenizer, schema, question)
        
        print(f"\nGenerated SQL: {sql}")
        
        assert "SELECT" in sql.upper()
        assert "COUNT" in sql.upper() or "count" in sql.lower()
        assert "GROUP BY" in sql.upper()
        assert "HAVING" in sql.upper() or "> 1" in sql
    
    def test_order_by_expression(self, expert_model):
        """Test ORDER BY with expression"""
        model, tokenizer = expert_model
        
        schema = "CREATE TABLE head (name VARCHAR, born_state VARCHAR, age VARCHAR)"
        question = "List name, born state and age of heads ordered by age"
        sql = generate_sql(model, tokenizer, schema, question)
        
        print(f"\nGenerated SQL: {sql}")
        
        assert "SELECT" in sql.upper()
        assert "ORDER BY" in sql.upper()
        assert "age" in sql.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])







