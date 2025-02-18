
import os
import json
import re

from test_runner_function import runner
techniques = ["fewshot", "self_planning", "zeroshot_cot"]
# technique = "fewshot_cot"
r_s1 = [ "kornia", "pyro"]
r_s2 = [  "scikit-learn", "neurodiffeq",  "umap", "vision", "small-text", "inference", "GPflow", "recommenders", "Laplace", "pyod", "pfrl", "pennylane", "nncf", "neupy", "emukit", "DeepReg", "deepchem", "cleanlab", "pytorch-forecasting", "pytorch-widedeep","torchgeo", "lightly"]
r_s3 = ["nlp-architecht", "imagededup","cleanlab", "pytorch-forecasting", "pytorch-widedeep","torchgeo", "lightly"]
r_s = ["pytorch3d"] + r_s1 + r_s2 + r_s3
remaining2 = ["litgpt" , "avalanche"]
llms = [  "openai"]
for technique in techniques:
    for llm in llms:
        for r in r_s:
            folder_path = f"../../results/llm-output/{technique}/output_{llm}/{r}"
            print(folder_path)
            if r != "pytorch3d":
                BASE_PATH = "/local/data0/moved_data/publishablew/"
            else:
                BASE_PATH = "/local/data0/moved_data/"
            # Loop through each file in the folder
            for filename in os.listdir(folder_path):
                # print("##################################")
                # print(filename)
                # print("##################################")
                if filename.endswith('.json') and not "processed_classes" in filename:
                    # if not "train" in filename:
                    #     continue
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, 'r') as f:
                        # Load the JSON data
                        # if  not "get_sobel_kernel2d_2nd_order" in file_path:
                        #     continue
                        data = json.load(f)
                        repo = f"{r}"
                        p = data["ground_truth"].split("#")[0]
                        f_name = data["function_name"]
                        prompt = data["prompt"]
                        tests = data["test"]
                        result = data.get('result', '')
                        stage = data.get('stage', '')
                        task = data.get('task', '')
                        data = data.get('data', '')
                        print(result)
                        # exit(1)
                        if repo != "pytorch3d":
                            path_to_fn = os.path.join(BASE_PATH, repo, repo, p)
                            tests = os.path.join(BASE_PATH, repo, repo, tests)
                        else:
                            path_to_fn = os.path.join(BASE_PATH, repo, p)
                            tests = os.path.join(BASE_PATH,repo, tests)
                        # Use regex to extract the code starting from '```python\n'
                        match = re.search(r'```python\n(.*?)(?:\n```|$)', result, re.DOTALL)
                        if match:
                            code = match.group(1)

                            code = code.split("# Example usage")
                            code = code[0]
                            # exit(1)
                            # exit(1)
                            with open("t.txt", "w") as f:
                                f.write(code)
                            test_result = runner(path_to_fn, f_name, tests, filename, llm, r, code, prompt, technique)
                            
                        elif re.search(r'```\n(.*?)(?:\n```|$)', result, re.DOTALL):
                            match1 = re.search(r'```\n(.*?)(?:\n```|$)', result, re.DOTALL)
                            if match1:
                                code = match1.group(1)
                                print(code)
                                with open("t.txt", "w") as f:
                                    f.write(code)
                                    print(code)
                                test_result = runner(path_to_fn, f_name, tests, filename, llm, r, code, prompt, technique)
                        else:
                            if "def" in result:
                            
                                code = result
                                print(code)
                                with open("t.txt", "w") as f:
                                    f.write(code)
                                    print(code)
                                test_result = runner(path_to_fn, f_name, tests, filename, llm, r, code, prompt, technique)
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
                            
                            
                        
