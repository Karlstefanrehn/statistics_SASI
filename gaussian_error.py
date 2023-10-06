
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



def Gaussioan_stats_analysis():
    
    
        #https://www.geeksforgeeks.org/probabilistic-predictions-with-gaussian-process-classification-gpc-in-scikit-learn/

        #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2677737/
        
        
            
        ''' Below  the Gaussian process classification'''
        
        
        
        from scipy.stats import norm
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

       
        df = pd.read_excel(r'/Users/rehnan/Library/CloudStorage/OneDrive-Chalmers/Projektet/bridge_data_/csv_2022/raw_data/carbon_plot_1607st_nov16_second_VERSION_7limit_both.xlsx')

        
        # Define a function to determine the p-value from GP predictions and actual 2017 data
        def compute_p_value(y_2017, y_pred, sigma):
            # Assuming the differences are normally distributed, we can compute p-values using the CDF
            diff = y_2017 - y_pred
            standardized_diff = diff / sigma
            p_values = 2 * (1 - norm.cdf(np.abs(standardized_diff)))  # Two-tailed p-value
            return np.mean(p_values)  # Return the average p-value for the group

        results = []
        groups = df.groupby('PO8')
        for name, group in groups:
            x_2007 = group.index.to_numpy().reshape(-1, 1)
            y_2007 = group['C1'].to_numpy()
            
            x_2017 = group.index.to_numpy().reshape(-1, 1)
            y_2017 = group['C2'].to_numpy()

            # GP fit for 2007 data
            kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
            gp.fit(x_2007, y_2007)
            y_pred, sigma = gp.predict(x_2017, return_std=True)

            # Calculate p-value
            p_value = compute_p_value(y_2017, y_pred, sigma)
            
            # Store results
            results.append({
                'PO8': name,
                'p-value': p_value,
                'Statistically Significant': 'Yes' if p_value < 0.05 else 'No'
            })

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        print(results_df)


Gaussioan_stats_analysis()