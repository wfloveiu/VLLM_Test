# # import torch
# # import numpy as np
# import pandas as pd
# import os

# def write_tensor_to_excel(tensor, output_dir, layer, filename_prefix):
    
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     excel_filename = os.path.join(output_dir, f"{filename_prefix}_{layer}.xlsx")
    
#     array = tensor.cpu().numpy().T
    
#     if os.path.exists(excel_filename):
#         df_existing = pd.read_excel(excel_filename, index_col=0)
#     else:
#         df_existing = pd.DataFrame()
        
#     start_col_index = len(df_existing.columns) + 1
#     columns = [f'Column {i}' for i in range(start_col_index, start_col_index + array.shape[1])]
#     df_new = pd.DataFrame(array, columns=columns)
    
#     df_combined = pd.concat([df_existing, df_new], axis=1)
#     df_combined.to_excel(excel_filename)
    
#     print(f"Data has been written to {excel_filename}")


import pandas as pd
import os

def write_tensor(tensor, output_dir, layer, filename_prefix):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csv_filename = os.path.join(output_dir, f"{filename_prefix}_{layer}.csv")
    array = tensor.cpu().numpy().T

    if os.path.exists(csv_filename):
        df_existing = pd.read_csv(csv_filename, index_col=0)
    else:
        df_existing = pd.DataFrame()
        
    start_col_index = len(df_existing.columns) + 1
    columns = [f'Column {i}' for i in range(start_col_index, start_col_index + array.shape[1])] 
    df_new = pd.DataFrame(array, columns=columns)
    df_combined = pd.concat([df_existing, df_new], axis=1)
    df_combined.to_csv(csv_filename)
    
    # print(f"Data has been written to {csv_filename}")
