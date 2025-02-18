import boto3
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from botocore.exceptions import ClientError

from openai import OpenAI
import csv
import json
API_KEY = "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"
Geminay_KEY = "AIzaSyDckIvqw74F_C4F_7qI0dxOIFR3jnJdhGs"
from prompts.few_shot_cot import cot_fewshot
from prompts.zero_shot_cot import ZeroShotCoT
from prompts.few_shot import fewshot
zero_shot_cot = ZeroShotCoT("dlbench", "4o", "zero_cot")
client = OpenAI(api_key=API_KEY)
# import google.generativeai as genai

# genai.configure(api_key=Geminay_KEY)

def call_geminai(prompt):
    return
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    print(response)
    try:
        return response.text
    except Exception as e:
        return str(e)
def get_completion(prompt,tem = 0 , model="gpt-4o"):
    try:
        c=[
                {'role': 'user', 'content': prompt}]


        response = client.chat.completions.create(model=model,
        messages=c,
        max_tokens=10000,
        temperature=0)

        return response.choices[0].message.content
    except:
        return "Error"
def call_mistral(prompt):
    from botocore.exceptions import ClientError

    # Create a Bedrock Runtime client in the AWS Region of your choice.
    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    # Set the model ID, e.g., Mistral Large.
    model_id = "mistral.mistral-7b-instruct-v0:2"

    # Define the prompt for the model.

    # Embed the prompt in Mistral's instruction format.
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"

    # Format the request payload using the model's native structure.
    native_request = {
        "prompt": formatted_prompt,
        "max_tokens": 2048,
        "temperature": 0.1,
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    try:
        # Invoke the model with the request.
        response = client.invoke_model(modelId=model_id, body=request)

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)

    # Decode the response body.
    model_response = json.loads(response["body"].read())

    # Extract and print the response text.
    response_text = model_response["outputs"][0]["text"]
    return response_text
def call_llama(prompt):
    from botocore.exceptions import ClientError

    # Create a Bedrock Runtime client in the AWS Region of your choice.
    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    model_id = "meta.llama3-1-70b-instruct-v1:0"
    formatted_prompt = f"""
<|begin_of_text|>
<|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# Format the request payload using the model's native structure.
    native_request = {
        "prompt": formatted_prompt,
        "max_gen_len": 2048,
        "temperature": 0.1,
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    try:
        # Invoke the model with the request.
        response = client.invoke_model(modelId=model_id, body=request)

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")

    # Decode the response body.
    model_response = json.loads(response["body"].read())

    # Extract and print the response text.
    response_text = model_response["generation"]
    return response_text
def call_antropic(prompt):
# Create a Bedrock Runtime client in the AWS Region of your choice.

    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    # Set the model ID, e.g., Claude 3 Haiku.
    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

    # Define the prompt for the model.
    prompt += "\nONLY RETURN THE CODE"
    # Format the request payload using the model's native structure.
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8000,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    try:
        # Invoke the model with the request.
        response = client.invoke_model(modelId=model_id, body=request)

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)

    # Decode the response body.
    model_response = json.loads(response["body"].read())

    # Extract and print the response text.
    response_text = model_response["content"][0]["text"]
    return response_text


import os
import json

def process_file_data(prompt, llm):
    
    if llm == "antropic":
        return call_antropic(prompt)
    if llm == "mistral":
        return call_mistral(prompt)
    if llm == "llama":
        return call_llama(prompt=prompt)
    if llm == "openai":
        return get_completion(prompt=prompt)
    if llm == "geminai":
        return call_geminai(prompt=prompt)

import json
import re

def parse_text_to_json(text):
    lines = text.split('\n')
    result = {}
    current_key = None
    current_value_lines = []
    valid_keys = {
        'stage',
        'task',
        'data',
        'prompt',
        'ground truth',
        'repo',
        'function',
        'test_cases',
        'class'
    }
    
    for line in lines:
        stripped_line = line.strip()
        
        if not stripped_line:
            if current_key is not None:
                current_value_lines.append('')
            continue
        
        key_match = re.match(r'^(\s*)(.*?):\s*(.*)$', line)
        
        if key_match:
            key = key_match.group(2).strip().lower()
            value = key_match.group(3)
            if key in valid_keys:
                if current_key is not None:
                    result[current_key] = '\n'.join(current_value_lines).strip()
                current_key = key_match.group(2).strip()
                current_value_lines = [value] if value else []
                continue  
        if current_key is not None:
            current_value_lines.append(line)
    if current_key is not None:
        result[current_key] = '\n'.join(current_value_lines).strip()
    return result

from tqdm import tqdm
def process_txt(input_file_path):

    with open(input_file_path, 'r') as f:
        file_content = f.read()

    parsed_data = parse_text_to_json(file_content)
    
    return parsed_data
from tqdm import tqdm
def process_folder(input_folder, output_folder, llm, repo, technique):
    """
    Process each text file in the input folder, parse it to JSON, call the process_file_data function,
    and save the results into an output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    counter = 0
    
    for filename in tqdm(os.listdir(input_folder)):
        if not f"{repo}" in filename:
            continue
        input_file_path = os.path.join(input_folder, filename)
        counter += 1

        if os.path.isfile(input_file_path) and filename.endswith(".txt"):
            

            parsed_data = process_txt(input_file_path)

            
            prompt = parsed_data ["prompt"]
            new_prompt = prompt
            if technique == "fewshot_cot":
                print(technique)
                new_prompt = cot_fewshot(prompt)
            if technique == "zeroshot_cot":
                print(technique)
                new_prompt = zero_shot_cot.form_technique_prompt(prompt)
            if technique == "fewshot":
                print(technique)
                new_prompt = fewshot(prompt)
            
            if technique == "self_planning":
                from prompts.self_planning import self_planning_implementation, self_planning_plan
                p = self_planning_plan(prompt=prompt)
                if llm == "openai":
                    plan = get_completion(prompt=p)
                if llm == "antropic":
                    plan = call_antropic(prompt=p)
                
                new_prompt = self_planning_implementation(prompt, plan)
            result = process_file_data(new_prompt, llm)
            if  "function" in parsed_data.keys():
                
                function_name = parsed_data ["function"]
            elif "f_name" in parsed_data.keys(): 
                function_name = parsed_data ["f_name"]
            else:
                function_name = ""
            if "class" in parsed_data.keys():
                class_name = parsed_data["class"]
            else:
                class_name = ""
                
            ground_truth = parsed_data["ground Truth"]
            test = parsed_data["test_cases"]
            output_file_path = os.path.join(output_folder, f"processed_{filename.replace('.txt', '.json')}")
            output_data = {
                "result": result,
                "prompt": prompt,
                "function_name": function_name,
                "ground_truth": ground_truth,
                "test": test,
                "class":class_name,
                "stage":parsed_data["stage"],
                "task": parsed_data["task"],
                "data" : parsed_data["data"]
            }
            output_dir = os.path.dirname(output_file_path)
            os.makedirs(output_dir, exist_ok=True)
            with open(output_file_path, 'w') as out_f:
                json.dump(output_data, out_f, indent=4)

            print(f"Processed and saved result for {filename}")

def call():
    llms = ["antropic"]
    repos = list(set([ "kornia", "scikit-learn", "imagededup", "neurodiffeq", "pytorch3d", "umap", "vision", "small-text", "inference", "GPflow", "recommenders", "Laplace", "pyro", "pyod", "pfrl", "pennylane", "nncf", "neupy", "litgpt" , "emukit", "DeepReg", "deepchem", "cleanlab", "pytorch-forecasting", "pytorch-widedeep", "avalanche", "nlp-architecht","torchgeo", "lightly", "emukit"]))
    print(len(repos))
    technique = "self_planning"
    for llm in llms:
        for repo in repos:
            input_folder = "../../data/DLEval-20240920T201632Z-001/DLEval"
            output_folder = f"../../results/llm-output/{technique}/output_{llm}//{repo}"

            process_folder(input_folder, output_folder, llm , repo, technique)
call()
