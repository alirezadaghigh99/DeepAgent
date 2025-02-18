
import os
import json
import re
from test_runner_class import runner
technique = "fewshot"
techniques = [ "self_planning", "zeroshot_cot"]

r_s = [ "kornia", "pytorch3d"]
r_s2 = [  "scikit-learn", "neurodiffeq",  "umap", "vision", "small-text", "GPflow", "recommenders", "Laplace", "pyod", "pfrl", "pennylane", "nncf", "neupy", "emukit", "DeepReg", "deepchem", "cleanlab", "pytorch-forecasting", "pytorch-widedeep","torchgeo", "lightly"]
r_s3 = ["nlp-architecht", "imagededup","cleanlab", "pytorch-forecasting", "pytorch-widedeep","torchgeo", "lightly"]
# r_s =   r_s2 + r_s3
llms = [ "openai"]

for technique in techniques:
    for r in r_s:
            
        for llm in llms:
            # original_data_params = f"LLMs/output_mistral/{r}/"
            folder_path = f"../../results/llm-output/{technique}/output_{llm}/{r}"
            # BASE_PATH = "/home/aliredaq/Desktop/CG-DeepLearning/CGBench/repo_test_v4/"
            if r != "pytorch3d":
                BASE_PATH = "/local/data0/moved_data/publishablew/"
            else:
                
                BASE_PATH = "/local/data0/moved_data/"
            # BASE_PATH = "/local/data0/moved_data/"
            # Loop through each file in the folder
            for filename in os.listdir(folder_path):
                print("##################################")
                print(filename)
                print("##################################")
                if filename.endswith('.json') and "processed_class" in filename:
                    if "korniaforward97" in filename or "korniaforward97" in filename:
                        continue
                    file_path = os.path.join(folder_path, filename)
                    original_param_path = os.path.join(folder_path, filename)
                    print(original_param_path)
                    with open(original_param_path, "r") as f:
                        data = json.load(f)
                        class_name = data["class"]
                        # if "Translate" not in class_name:
                        #     continue
                        p = data["ground_truth"].split("#")[0]
                        f_name = data["function_name"]
                        tests = data["test"]
                        prompt = data["prompt"]
                        stage = data.get('stage', '')
                        task = data.get('task', '')
                        data = data.get('data', '')
                        repo = f"{r}"
                        if repo != "pytorch3d":
                            path_to_fn = os.path.join(BASE_PATH, repo, repo, p)
                            tests = os.path.join(BASE_PATH, repo, repo, tests)
                        else:
                            path_to_fn = os.path.join(BASE_PATH, repo, p)
                            tests = os.path.join(BASE_PATH,repo, tests)
                        print(path_to_fn)
                        print(filename)
                        
                    with open(file_path, 'r') as f:
                        # Load the JSON data
                        data = json.load(f)
                        
                        
                        result = data.get('result', '')
                        
                        
                        # Use regex to extract the code starting from '```python\n'
                        match = re.search(r'```python\n(.*?)(?:\n```|$)', result, re.DOTALL)
                        if match:
                            code = match.group(1)
                            code = code.split("# Example usage")
                            code = code[0]
                            print(code)
                            
                            with open("t.txt", "w") as f:
                                f.write(code)
                                print(code)
                            test_result = runner(file_path=path_to_fn, function_name=f_name, test_file=tests, class_name=class_name, code_str=code, llm_output=filename, repository = r, llm = llm, prompt=prompt, technique=technique)
                        elif re.search(r'```\n(.*?)(?:\n```|$)', result, re.DOTALL):
                            match1 = re.search(r'```\n(.*?)(?:\n```|$)', result, re.DOTALL)
                            if match1:
                                code = match1.group(1)
                                print(code)
                                with open("t.txt", "w") as f:
                                    f.write(code)
                                    print(code)
                                test_result = runner(file_path=path_to_fn, function_name=f_name, test_file=tests,class_name=class_name, code_str=code, llm_output=filename, repository = r, llm = llm, prompt=prompt, technique=technique)
                        else:
                            if "def" in result:
                                code = result
                                with open("t.txt", "w") as f:
                                    f.write(code)
                                    print(code)
                                test_result = runner(file_path=path_to_fn, function_name=f_name, test_file=tests,class_name=class_name, code_str=code, llm_output=filename, repository = r, llm = llm, prompt=prompt, technique=technique)
                            else:
                                print(f"No code found in {filename}")
                    data_to_save = {
                            "test_result" :test_result, 
                            "file_path" : filename,
                            "stage": stage,
                            "task": task, 
                            "data": data
                        }
                    with open(f"result_{technique}.jsonl", "a") as f: 
                        json.dump(data_to_save, f)
                        f.write("\n")
                            
                        
