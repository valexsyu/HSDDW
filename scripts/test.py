# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# def load_and_process_csv(file_path):
#     data = []
#     max_length = 0
    
#     # Manually read the CSV file, handling rows with different lengths
#     with open(file_path, 'r') as file:
#         for line in file:
#             row = list(map(float, line.strip().split(',')))
#             data.append(row)
#             max_length = max(max_length, len(row))
    
#     # Pad each row with NaN values to make them all the same length
#     for row in data:
#         if len(row) < max_length:
#             row.extend([np.nan] * (max_length - len(row)))

#     # Convert the list of lists into a DataFrame
#     df = pd.DataFrame(data)

#     # Calculate mean and variance for each column, ignoring NaN values
#     mean_values = df.mean(skipna=True)
#     variance_values = df.var(skipna=True)
    
#     return mean_values, variance_values

# def plot_multiple_datasets(root_path, dir_paths, file_names, save_path):
#     plt.figure(figsize=(10, 6))
    
#     # Different colors for different datasets
#     colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']
    
#     for i, (dir_path, file_name) in enumerate(zip(dir_paths, file_names)):
#         file_path = root_path + dir_path + file_name
#         mean_values, variance_values = load_and_process_csv(file_path)
        
#         # # Plot mean
#         # plt.plot(mean_values, marker='o', color=colors[i % len(colors)], label=f'{file_name} Mean')

#         # Plot mean
#         plt.plot(mean_values, color=colors[i % len(colors)], label=f'{file_name} Mean')        
#         # Fill between mean ± variance
#         plt.fill_between(range(len(mean_values)), mean_values - variance_values, 
#                          mean_values + variance_values, color=colors[i % len(colors)], alpha=0.3, 
#                          label=f'{file_name} Mean ± Variance')
    
#     plt.title('Mean and Variance Comparison Across Datasets')
#     plt.xlabel('Time Step')
#     plt.ylabel('Entropy Value')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(save_path)

# # Example usage
# root_path = '/work/valex1377/LLMSpeculativeSampling/experiments/'
# dir_paths = ['7b_fp16_4096tok_recode_only/', '68m_fp16_4096tok_recode_only/', '70b_fp16_4096tok_recode_only/']  # List of directories
# file_names = [
#     'entropy_mesure_accepted_entropy_7b_fp16_4096tok_recode_only.csv', 
#     'entropy_mesure_accepted_entropy_68m_fp16_4096tok_recode_only.csv',
#     'entropy_mesure_accepted_entropy_70b_fp16_4096tok_recode_only.csv'
# ]  # List of file names

# save_path = root_path + 'comparison_entropy.jpg'
# plot_multiple_datasets(root_path, dir_paths, file_names, save_path)














import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_process_csv(file_path):
    data = []
    max_length = 0
    
    # Manually read the CSV file, handling rows with different lengths
    with open(file_path, 'r') as file:
        for line in file:
            row = list(map(float, line.strip().split(',')))
            data.append(row)
            max_length = max(max_length, len(row))
    
    # Pad each row with NaN values to make them all the same length
    for row in data:
        if len(row) < max_length:
            row.extend([np.nan] * (max_length - len(row)))

    # Convert the list of lists into a DataFrame
    df = pd.DataFrame(data)

    # Calculate mean and variance for each column, ignoring NaN values
    mean_values = df.mean(skipna=True)
    variance_values = df.var(skipna=True)
    
    return mean_values, variance_values

def load_length_data(file_path):
    # Load the step lengths into a pandas Series
    length_data = pd.read_csv(file_path, header=None).squeeze("columns")
    return length_data

def plot_multiple_datasets_with_lengths(root_path, dir_paths, file_names, length_file_paths, save_path):
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']
    
    for i, (dir_path, file_name, length_file_path) in enumerate(zip(dir_paths, file_names, length_file_paths)):
        file_path = root_path + dir_path + file_name
        length_file_full_path = root_path + dir_path + length_file_path
        
        # Load entropy data
        mean_values, variance_values = load_and_process_csv(file_path)
        
        # Load length data
        length_data = load_length_data(length_file_full_path)
        
        # Plot mean and variance of entropy
        plt.subplot(2, 1, 1)
        plt.plot(mean_values, marker='o', color=colors[i % len(colors)], label=f'{file_name} Mean')
        # plt.fill_between(range(len(mean_values)), mean_values - variance_values, 
        #                  mean_values + variance_values, color=colors[i % len(colors)], alpha=0.3, 
        #                  label=f'{file_name} Mean ± Variance')
        
        # Plot length data as time series
        plt.subplot(2, 1, 2)
        plt.plot(range(1, len(length_data) + 1), length_data, marker='o', color=colors[i % len(colors)], 
                 label=f'{file_name} Length per Data Point')
    
    # Finalize the entropy plot
    plt.subplot(2, 1, 1)
    plt.title('Mean and Variance Comparison Across Datasets')
    plt.xlabel('Time Step')
    plt.ylabel('Entropy Value')
    plt.legend()
    plt.grid(True)

    # Finalize the length distribution plot
    plt.subplot(2, 1, 2)
    plt.title('Length of Each Data Point Across Datasets')
    plt.xlabel('Data Point Index')
    plt.ylabel('Length')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)

# Example usage
root_path = '/work/valex1377/LLMSpeculativeSampling/experiments/'
dir_paths = [
    '7b_fp16_4096tok_recode_only/', 
    # '68m_fp16_4096tok_recode_only/', 
    # '70b_fp16_4096tok_recode_only/'
]  # List of directories
file_names = [
    'entropy_mesure_accepted_entropy_7b_fp16_4096tok_recode_only.csv', 
    # 'entropy_mesure_accepted_entropy_68m_fp16_4096tok_recode_only.csv',
    # 'entropy_mesure_accepted_entropy_70b_fp16_4096tok_recode_only.csv',
]  # List of file names

# Corresponding length files for each dataset
length_file_paths = [
    'entropy_mesure_output_length_7b_fp16_4096tok_recode_only.csv', 
    # 'entropy_mesure_output_length_68m_fp16_4096tok_recode_only.csv',
    # 'entropy_mesure_output_length_70b_fp16_4096tok_recode_only.csv',
]

save_path = root_path + 'comparison_entropy_with_lengths.jpg'
plot_multiple_datasets_with_lengths(root_path, dir_paths, file_names, length_file_paths, save_path)


