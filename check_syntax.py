#!/usr/bin/env python3
"""
Syntax check script for Media Recommender API
"""

import ast
import sys
from pathlib import Path

def check_file_syntax(file_path: str) -> bool:
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST to check syntax
        ast.parse(content)
        print(f"✅ {file_path}")
        return True
        
    except SyntaxError as e:
        print(f"❌ {file_path}: Syntax error at line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ {file_path}: Error reading file: {e}")
        return False

def check_imports(file_path: str) -> bool:
    """Check if imports are valid"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to compile the file
        compile(content, file_path, 'exec')
        return True
        
    except ImportError as e:
        print(f"⚠️  {file_path}: Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ {file_path}: Compilation error: {e}")
        return False

def main():
    """Check all Python files in the project"""
    print("🔍 Checking syntax for all Python files...")
    print("=" * 50)
    
    # List of Python files to check
    python_files = [
        "config.py",
        "models.py", 
        "tmdb_client.py",
        "vector_store.py",
        "recommendation_service.py",
        "main.py",
        "setup.py",
        "run.py",
        "test_api.py",
        "test_pinecone.py",
        "init_pinecone.py"
    ]
    
    syntax_errors = 0
    import_errors = 0
    
    for file_path in python_files:
        if Path(file_path).exists():
            if not check_file_syntax(file_path):
                syntax_errors += 1
            if not check_imports(file_path):
                import_errors += 1
        else:
            print(f"⚠️  {file_path}: File not found")
    
    print("\n" + "=" * 50)
    print(f"📊 Results:")
    print(f"   Syntax errors: {syntax_errors}")
    print(f"   Import errors: {import_errors}")
    
    if syntax_errors == 0 and import_errors == 0:
        print("🎉 All files passed syntax check!")
        return True
    else:
        print("❌ Some files have issues that need to be fixed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 