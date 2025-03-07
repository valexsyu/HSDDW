import json
import re

# File paths
#text_file_path = "/work/valex1377/LLMSpeculativeSampling/experiments/mt_bench/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_temp1e-20_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_generated_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv"

text_file_path = "/work/valex1377/LLMSpeculativeSampling/experiments/mt_bench/sp_dy_gamma_10_70b_7b_68m_adv_topkp0_temp1_fp16_2048tok/sp_dy_gamma_etp_hrchl_adv_generated_sp_dy_gamma_10_70b_7b_68m_adv_topkp0_fp16_2048tok.csv"
json_file_path = "/work/valex1377/LLMSpeculativeSampling/my_datasets/mt_bench/question.json"
output_file_path = "/work/valex1377/LLMSpeculativeSampling/my_datasets/mt_bench/question_multi.json"

# # Step 1: Parse the text file
# with open(text_file_path, "r", encoding="utf-8") as text_file:
#     text_data = text_file.read().split("[_START_]")[1:]  # Split and remove the first empty entry
#     text_entries = [entry.strip() for entry in text_data]  # Remove leading/trailing spaces


with open(text_file_path, "r", encoding="utf-8") as text_file:
    text_data = text_file.read()
    text_entries = re.split(r"\[_START_\d+_\]", text_data)[1:]  # Split by [_START_x_] and remove the first empty entry
    text_entries = [entry.strip() for entry in text_entries]  # Remove leading/trailing spaces



# Step 2: Parse the JSON file
with open(json_file_path, "r", encoding="utf-8") as json_file:
    json_entries = [json.loads(line.strip()) for line in json_file]

# Step 3: Combine the entries
combined_entries = []
for text_entry, json_entry in zip(text_entries, json_entries):
    combined_entry = json_entry
    combined_entry['turns'][0] = combined_entry['turns'][0] + '\n\n' + text_entry + '\n\n' + json_entry['turns'][1]
    combined_entries.append(combined_entry)

# Step 4: Write to output.json
with open(output_file_path, "w", encoding="utf-8") as output_file:
    json.dump(combined_entries, output_file, ensure_ascii=False, indent=4)

print(f"Output written to {output_file_path}")
