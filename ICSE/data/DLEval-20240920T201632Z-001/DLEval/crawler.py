import re
def parse_details(file_path):
    """
    Parse the details from a text file and extract specific fields: 'function', 'repo', 'ground Truth', and optionally 'class'.

    Args:
        file_path (str): Path to the text file containing details.

    Returns:
        dict: A dictionary containing the parsed fields.
    """
    # Define regex patterns for the important fields
    patterns = {
        "function": r"function:\s*(\w+)",
        "repo": r"repo:\s*([\w\-/]+)",
        "ground Truth": r"ground Truth:\s*([\w/\-\.]+\.py)",
        "class": r"class:\s*(\w+)"
    }

    # Initialize the result dictionary
    details = {}

    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()

    # Apply regex patterns to extract fields
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            details[key] = match.group(1)

    # Return only the required fields
    return {key: details[key] for key in ["ground Truth", "function", "class", "repo"] if key in details}


# Example usage

import os

from function_return import crawl_function_definition
files = os.listdir(".")
base_path = "/local/data0/moved_data/publishablew"
registry = "/home/aliredaq/Desktop/codes/"
data = []
for file_path in files:
    if "txt" in file_path and "388" in file_path:
        parsed_details = parse_details(file_path)
        print(parsed_details)
        repository = parsed_details["repo"]
        if "function" not in parsed_details or "deepcheck" in repository or "avalanc" in repository:
            continue
        function = parsed_details["function"]
        file_p = parsed_details["ground Truth"]
        
        if "class" in parsed_details:
            class_name = parsed_details["class"]
        else:
            class_name = None
    else:
        continue
    file_p = os.path.join(base_path, repository, repository, file_p)
    code = crawl_function_definition(file_path=file_p, function_name=function, class_name=class_name)
    print(code)
    with open(registry+file_path, "w") as f:
        f.write(code)
