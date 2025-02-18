import os
import ast
import pandas as pd

def count_physical_loc(code_string):
    """
    Count the physical lines of code, excluding comments and docstrings.
    """
    lines = remove_comments_and_docstrings(code_string).split('\n')
    non_empty_lines = [line for line in lines if line.strip() != '']
    return len(non_empty_lines)

def calculate_cyclomatic_complexity(code):
    """
    Calculate cyclomatic complexity based on the number of decision points.
    """
    parsed_code = ast.parse(remove_comments_and_docstrings(code))
    complexity = 1  # Start with 1 for the default path

    for node in ast.walk(parsed_code):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.Try, ast.With)):  
            complexity += 1
        elif isinstance(node, ast.FunctionDef):
            complexity += len(node.args.args)  # Add complexity for arguments

    return complexity

def calculate_halstead_complexity(code):
    """
    Calculate Halstead complexity metrics (e.g., vocabulary).
    """
    parsed_code = ast.parse(remove_comments_and_docstrings(code))
    operators = set(['+', '-', '*', '/', '%', '**', '//', '=', '==', '!=', '<', '>', '<=', '>=', 'and', 'or', 'not', 'is', 'in'])
    operands = set()

    operator_count = 0
    operand_count = 0

    for node in ast.walk(parsed_code):
        if isinstance(node, ast.BinOp):
            operator_count += 1
            operands.add(node.op.__class__.__name__)
        elif isinstance(node, (ast.Name, ast.Constant)):
            operand_count += 1
            operands.add(getattr(node, 'id', repr(node)))

    vocabulary = len(operators) + len(operands)
    return vocabulary

def calculate_mi(code_string):
    """
    Estimate the Maintainability Index using a simplified approach.
    """
    loc = count_physical_loc(code_string)
    cc = calculate_cyclomatic_complexity(code_string)
    vocab = calculate_halstead_complexity(code_string)
    
    # Simplified Maintainability Index formula
    mi = max(0, 171 - 5.2 * (vocab / max(loc, 1)) - 0.23 * cc - 16.2 * loc)
    return mi

def calculate_cognitive_complexity(code):
    """
    Calculate Cognitive Complexity based on AST traversal.
    """
    parsed_code = ast.parse(remove_comments_and_docstrings(code))
    complexity = 0

    def traverse(node, nesting=0):
        nonlocal complexity
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With)):
                complexity += 1 + nesting  # Add for nesting level
                traverse(child, nesting + 1)
            else:
                traverse(child, nesting)

    traverse(parsed_code)
    return complexity

def remove_comments_and_docstrings(code):
    """
    Remove comments and docstrings from Python code.
    """
    try:
        parsed_code = ast.parse(code)
        for node in ast.walk(parsed_code):
            if isinstance(node, ast.FunctionDef):
                node.body = [stmt for stmt in node.body if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Str))]
        return ast.unparse(parsed_code)
    except Exception:
        # Fallback: Remove comments manually
        lines = code.split('\n')
        clean_lines = []
        for line in lines:
            line = line.split('#')[0].strip()  # Remove inline comments
            if line:  # Skip empty lines
                clean_lines.append(line)
        return '\n'.join(clean_lines)

# Directory containing the files
directory_path = "/home/aliredaq/Desktop/codes/"
results = []

# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    print(filename)
    file_path = os.path.join(directory_path, filename)
    if filename.endswith(".txt") and os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            code = file.read()
            if code.strip():  # Process non-empty files
                results.append({
                    "Filename": filename,
                    "Physical LOC": count_physical_loc(code),
                    "Cyclomatic Complexity": calculate_cyclomatic_complexity(code),
                    "Halstead Complexity": calculate_halstead_complexity(code),
                    "Maintainability Index": calculate_mi(code),
                    "Cognitive Complexity": calculate_cognitive_complexity(code)
                })

# Create a DataFrame and save it to CSV
df = pd.DataFrame(results)

# Calculate the mean of each metric
mean_metrics = df.drop(columns=["Filename"]).mean().reset_index()
mean_metrics.columns = ["Metric", "Average Value"]

# Save the mean metrics to a CSV
output_csv_path = "mean_metrics_summary1.csv"
mean_metrics.to_csv(output_csv_path, index=False)