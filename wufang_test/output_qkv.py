# import torch
# import numpy as np
import pandas as pd
import os

def write_tensor_to_excel(tensor, excel_filename):
    
    array = tensor.numpy().T
    
    if os.path.exists(excel_filename):
        df_existing = pd.read_excel(excel_filename, index_col=0)
    else:
        df_existing = pd.DataFrame()
    df_new = pd.DataFrame(array, columns=[f'Column {len(df_existing.columns) + 1}'])
    df_combined = pd.concat([df_existing, df_new], axis=1)
    df_combined.to_excel(excel_filename)
    
    print(f"Data has been written to {excel_filename}")


