import os

def not_processed(d1, d2):
    d1_s = []
    for root, dirs, files in os.walk(d1):
        for file in files:
            if not file.endswith('txt0'):
                d1_s.append(file.split(".json")[0])
    d2_s = []
    for root, dirs, files in os.walk(d2):
        for file in files:
            if not file.endswith('txt0'):
                if str(file).startswith("class_"):
                    file = file.replace("class_", "")
                d2_s.append(file.split(".txt")[0])
    print(d1_s)
    print(d2_s)
    for ds in d1_s:
        if ds not in d2_s:
            print(ds)

def count_files(directory):
    count = 0
    all_files = []
    for root, dirs, files in os.walk(directory):
        
        for file in files:
            all_files.append(file)
            count += 1
    return count, all_files
p = ""
# Specify the directory you want to search
data_path = "/home/aliredaq/Desktop/ICSE/data/DLEval-20240920T201632Z-001/DLEval"
datas = os.listdir(data_path)
directory_path_output_gemini = f'/home/aliredaq/Desktop/ICSE/results/llm-output/fewshot/output_openai'
# /home/aliredaq/Desktop/ICSE/results/llm-output/fewshot_cot
file_count_output_g, all_files = count_files(directory_path_output_gemini)
for data in datas:
    if data.endswith(".txt"):
        new_data = data.replace("txt", "json")
        new_data = "processed_" + new_data
        if new_data not in all_files:
            print(data)

# not_processed(directory_path_output_openai, directory_path_result_gemini)