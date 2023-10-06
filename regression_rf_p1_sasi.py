

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




def regression_and_rf_for_socclay():
    
    def clean_data(df):
        df = df.dropna()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        print(df)
        return df
   
    def compute_regression(df, dependent_var, predictors):
        X = df[predictors]
        y = df[dependent_var]
        model = LinearRegression().fit(X, y)
        
        # Fitting regression with statsmodels to get p-values and additional statistics
        X_with_const = sm.add_constant(X)
        sm_model = sm.OLS(y, X_with_const).fit()
        
        # Additional Statistics
        print("\nAdditional Regression Statistics:")
        print(f"R^2: {sm_model.rsquared:.4f}")
        print(f"F-statistic: {sm_model.fvalue:.4f}")
        print(f"Degrees of Freedom: {sm_model.df_model}, {sm_model.df_resid}")
        print(f"p-value (F-statistic): {sm_model.f_pvalue:.4f}")
        
        # Printing the p-values for each coefficient
        print("\nP-values for predictors:")
        for predictor, pval in zip(predictors, sm_model.pvalues[1:]):
            print(f"{predictor}: {pval:.4f}")

        y_pred = model.predict(X)
        r_sq = model.score(X, y)
        print(f"\ncoefficient of determination (from sklearn): {r_sq:.4f}")
        print(f"intercept: {model.intercept_}")
        print(f"coefficients: {model.coef_}")

        # Printing regression results in a more stylish format
        equation = f"{dependent_var} = {model.intercept_:.4f}"
        for i, coef in enumerate(model.coef_):
            equation += f" + {coef:.4f}*{predictors[i]}"
        
        print("\nRegression Equation:")
        print(equation)
        print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")

        return model, sm_model.pvalues[1:]

    def compute_random_forest_feature_importance(df, predictors, dependent_var):
        X_train, X_test, y_train, y_test = train_test_split(df[predictors], df[dependent_var], test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"\nRandom Forest MSE: {mse}")

        importances = rf.feature_importances_
        sorted_idx = importances.argsort()

        print("\nFeature Importances (from Random Forest):")
        for i in sorted_idx[::-1]:
            print(f"{predictors[i]}: {importances[i]}")
        
        return importances
    
    df_raw = pd.read_excel(file_path_here
    df = clean_data(df_raw)
    
    # List of predictors
    predictors = ['x','ph', 'ler', 'mat', 'map','share','n','C2']
    dependent_var = "socclay"
    
    model, p_values = compute_regression(df, dependent_var, predictors)
    importances = compute_random_forest_feature_importance(df, predictors, dependent_var)
    
    # Plotting regression equation
    plt.text(0.1, 0.5, f"Equation: {dependent_var} = {model.intercept_:.4f} + " + ' + '.join([f"{coef:.4f}*{predictor}" for coef, predictor in zip(model.coef_, predictors)]), wrap=True, fontsize=12, ha='left')
    plt.text(0.1, 0.3, f"R-squared: {model.score(df[predictors], df[dependent_var]):.4f}", wrap=True, fontsize=12, ha='left')
    plt.axis('off')
    plt.show()
    
    is_same = (df['socclay'] == df['C2']/df['ler']).all()
    print(is_same)


regression_and_rf_for_socclay()
