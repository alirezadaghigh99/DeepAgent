import os
import ast

def crawl_function_definition(file_path, function_name, class_name=None):
    """
    Crawl a Python file to extract the definition of a function. If a class name is provided,
    return the function definition within that class.

    Args:
        file_path (str): Path to the Python file to crawl.
        function_name (str): Name of the function to extract.
        class_name (str, optional): Name of the class containing the function.

    Returns:
        str: The definition of the function as a string, or an empty string if not found.
    """
    with open(file_path, 'r') as file:
        file_content = file.read()

    # Parse the Python file into an AST
    tree = ast.parse(file_content)

    # Get the lines of the file for precise slicing
    file_lines = file_content.splitlines()

    # Define a helper function to extract the source code of a node
    def get_source(node):
        start_line, start_col = node.lineno - 1, node.col_offset
        end_line, end_col = node.end_lineno - 1, node.end_col_offset
        
        if start_line == end_line:
            return file_lines[start_line][start_col:end_col]
        else:
            lines = [file_lines[start_line][start_col:]]
            lines.extend(file_lines[start_line + 1:end_line])
            lines.append(file_lines[end_line][:end_col])
            return '\n'.join(lines)

    # Traverse the AST to find the function
    for node in tree.body:
        # If class_name is provided, look for the class definition
        if isinstance(node, ast.ClassDef) and class_name and node.name == class_name:
            for class_node in node.body:
                if isinstance(class_node, ast.FunctionDef) and class_node.name == function_name:
                    return get_source(class_node)

        # If no class_name is provided, look for a top-level function
        if isinstance(node, ast.FunctionDef) and not class_name and node.name == function_name:
            return get_source(node)

    return ""

# Example usage
# repository = "pytorch-forecasting"
# base_path = "/local/data0/moved_data/publishablew"
# file_path = "pytorch_forecasting/data/encoders.py"  # Replace with the path to the Python file
# file_path = os.path.join(base_path, repository, repository, file_path)
# function_name = "transform"
# class_name = "GroupNormalizer"
# function_definition = crawl_function_definition(file_path, function_name, class_name)
# print(function_definition)

