def wilcoxon():
    ''''''
    import pandas as pd
    from scipy.stats import wilcoxon

    # Path to your Excel file
    file_path = "/file_path
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Function to perform the Wilcoxon signed-rank test and print the results
    def perform_wilcoxon_test(grouped_data, group_name):
        data_C2 = grouped_data['C2']
        data_C1 = grouped_data['C1']
        
        # Perform the Wilcoxon signed-rank test
        w, p = wilcoxon(data_C2, data_C1)

        # Display the results
        print(f"\nGroup: {group_name}")
        print(f"Wilcoxon test statistic: {w}")
        print(f"P-value: {p}")
        
        # Interpretation of results
        alpha = 0.05  # significance level
        if p > alpha:
            print('There is no significant difference between C2 and C1 (fail to reject H0)')
        else:
            print('There is a significant difference between C2 and C1 (reject H0)')

    # Perform the test for each group in 'PO8'
    unique_po8_values = df['PO8'].unique()
    for po8_value in unique_po8_values:
        group_data = df[df['PO8'] == po8_value]
        perform_wilcoxon_test(group_data, f"PO8 = {po8_value}")
        
    # Perform the test for each group in 'group'
    unique_group_values = df['group'].unique()
    for group_value in unique_group_values:
        group_data = df[df['group'] == group_value]
        perform_wilcoxon_test(group_data, f"group = {group_value}")




    

wilcoxon()
