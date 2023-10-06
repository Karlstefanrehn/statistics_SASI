def variance_po0():
    ''''''
    
    
    import pandas as pd
    from scipy.stats import ttest_ind
    import numpy as np

    # Path to your Excel file
    file_path = "/Users/rehnan/Library/CloudStorage/OneDrive-Chalmers/Projektet/bridge_data_/csv_2022/hypo_test/new_2022_june/for_po8_groupes/po8_u7.xlsx"

    # Read the Excel file
    # Note: If your data is in a particular sheet, you should specify it with, e.g., sheet_name='YourSheetName'
    df = pd.read_excel(file_path)

    # Basic descriptive statistics
    print("Descriptive statistics for 'C1':")
    print(df['C1'].describe())
    print("\nDescriptive statistics for 'C2':")
    print(df['C2'].describe())

    # Variance calculation
    print("\nVariance of 'c':", np.var(df['C1']))
    print("Variance of 'c_2':", np.var(df['C2']))

    # T-test
    t_stat, p_val = ttest_ind(df['C1'], df['C2'], equal_var=False, nan_policy='omit')  # 'omit' to ignore NaNs
    print("\nT-statistic:", t_stat)
    print("P-value:", p_val)

    # Check the significance
    alpha = 0.05  # Common significance level
    if p_val < alpha:
        print("The difference between 'c' and 'c_2' is statistically significant.")
    else:
        print("The difference between 'c' and 'c_2' is not statistically significant.")


    
    
variance_po0()