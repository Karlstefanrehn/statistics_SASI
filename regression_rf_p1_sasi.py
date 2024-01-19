


import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.stats import shapiro, boxcox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error

def regressionpart():

    def clean_data(df):
        df = df.dropna()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        print(df)
        return df
    
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df_filtered

    
    def transform_data(df, column):
        # Applying Reciprocal transformation
        # Ensure the column data does not contain zero as reciprocal of zero is undefined
        df_nonzero = df[df[column] != 0]
        df_nonzero[column] = df_nonzero[column]     # NU VISAR DENNA INGEN TRANFORMATION _ EFTERSOM DE INTE GJORDE SAKEN BÄTTRE
        return df_nonzero

    def compute_regression(df, dependent_var, predictors):
        X = df[predictors]
        y = df[dependent_var]
        model = LinearRegression().fit(X, y)
        
        # Fitting regression with statsmodels to get p-values and additional statistics
        X_with_const = sm.add_constant(X)
        sm_model = sm.OLS(y, X_with_const).fit()
        
        print("\nRegression Summary:")
        print(sm_model.summary())

        # Checking for Multicollinearity
        print("\nVariance Inflation Factor (VIF):")
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        print(vif_data)

        # Residual Analysis
        y_pred = model.predict(X)
        residuals = y - y_pred

        # Q-Q Plot for Normality of Residuals
        plt.figure(figsize=(10, 6))
        sm.qqplot(residuals, line ='45')
        plt.title('Q-Q Plot of Residuals')
        plt.show()

        # Shapiro-Wilk Test for Normality of Residuals
        shapiro_test = shapiro(residuals)
        print("\nShapiro-Wilk Test for Normality of Residuals:")
        print(f"Statistic: {shapiro_test.statistic:.4f}, p-value: {shapiro_test.pvalue:.4f}")

        # Plotting Residuals for Homoscedasticity
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals)
        plt.title('Residuals vs Fitted')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.axhline(y=0, color='red', linestyle='--')
        #plt.show()

        # Checking Normality of Residuals
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.title('Histogram of Residuals')
        plt.xlabel('Residuals')
        #lt.show()

        # Checking Linearity
        for p in predictors:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x=p, y=dependent_var)
            plt.title(f'Scatter plot of {p} vs {dependent_var}')
            plt.tight_layout()
            #plt.show()
        
        # Additional Statistics
        print("\nAdditional Regression Statistics:")
        print(f"R^2: {sm_model.rsquared:.4f}")
        print(f"F-statistic: {sm_model.fvalue:.4f}")
        print(f"Degrees of Freedom: {sm_model.df_model}, {sm_model.df_resid}")
        print(f"p-value (F-statistic): {sm_model.f_pvalue:.4f}")
        
        print("\nP-values for predictors:")
        for predictor, pval in zip(predictors, sm_model.pvalues[1:]):
            print(f"{predictor}: {pval:.4f}")

        y_pred = model.predict(X)
        r_sq = model.score(X, y)
        print(f"\ncoefficient of determination (from sklearn): {r_sq:.4f}")
        print(f"intercept: {model.intercept_}")
        print(f"coefficients: {model.coef_}")

        equation = f"{dependent_var} = {model.intercept_:.4f}"
        for i, coef in enumerate(model.coef_):
            equation += f" + {coef:.4f}*{predictors[i]}"
        
        print("\nRegression Equation:")
        print(equation)
        print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")

        return model, sm_model.pvalues[1:]

        # Load your data
    df = pd.read_excel(r'/Users/rehnan/Library/CloudStorage/OneDrive-Chalmers/Projektet/bridge_data_/csv_2022/raw_data/merged_file_for_regression_final4.xlsx')  # Replace with the actual path to your file

    
    # ANVÄNDER DENNA Spatial North  
    
    
    # Clean your data
    cleaned_df = clean_data(df)

    # Transform your dependent variable
    transformed_df = transform_data(cleaned_df, "C2")  # Assuming "C2" is the dependent variable

    # List of predictors
    predictors_reg = ['share', 'x', 'mat','map','ph','n', 'ler']  # Update this list as per your dataset
    
    # Remove outliers from the dependent variable
    cleaned_df_no_outliers = remove_outliers(cleaned_df, "C2")  # Replace "C2" with your dependent variable


    # Compute regression
    model, pvalues = compute_regression(transformed_df, "C2", predictors_reg)

# Execute the function
regressionpart()





def randomforestpart():
    
    
    def clean_data(df):
        df = df.dropna()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        #print(df)
        return df

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
        
        
        
        from sklearn.model_selection import cross_val_score, KFold    
        #Assuming df is your DataFrame with the correct columns.
        # Replace the dummy data with your actual data.
        df_2 = pd.DataFrame({
            'C2': df['C2'], 
            'share': df['share'], 
            'x': df['x'], 
            'ph': df['ph'], 
            'ler': df['ler'], 
            'mat': df['mat'], 
            'map': df['map'], 
            'n': df['n']
        })

        # Set the predictors and dependent variable
        predictors = ['x', 'ph', 'ler', 'mat', 'map', 'n', 'share']
        dependent_var = 'C2'

        # Initialize Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42)

        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mse_scores = cross_val_score(rf, df_2[predictors], df_2[dependent_var], scoring='neg_mean_squared_error', cv=kf)

        # Convert negative MSE scores to positive
        mse_scores = -mse_scores

        print(f"MSE scores for each fold: {mse_scores}")
        print(f"Average MSE across all folds: {np.mean(mse_scores)}")
        print(f"Standard Deviation of MSE across all folds: {np.std(mse_scores)}")
        
        return importances
    
    df_raw = pd.read_excel(r'/Users/rehnan/Library/CloudStorage/OneDrive-Chalmers/Projektet/bridge_data_/csv_2022/raw_data/merged_file_for_regression_final4.xlsx')
    df = clean_data(df_raw)
    
    #    /Users/rehnan/Library/CloudStorage/OneDrive-Chalmers/Projektet/Working Material/Papers/Paper 1/Nature Communications Submission/published data on Pangea/Data_share_Combining_article.xlsx
    
    # Set the predictors and dependent variable
    predictors = [ 'ph', 'mat',  'map',  'n', 'share'] #
    dependent_var = 'C2'
    

    importances = compute_random_forest_feature_importance(df, predictors, dependent_var)
    

    is_same = (df['socclay'] == df['C2']/df['ler']).all()
    print(is_same)
    
    
    
    # Assuming 'C2' and 'share' are columns in your DataFrame
    data = {
        'C2': df['C2'],  # Replace with your actual data
        'share': df['share']
    }

    df = pd.DataFrame(data)

    # Correlation Analysis
    correlation = df['C2'].corr(df['share'])
    print(f"Correlation between C2 and share: {correlation}")

    # Visual Inspection
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='C2', y='share')
    plt.title('Scatter Plot between C2 and share')
    plt.xlabel('C2')
    plt.ylabel('share')
    #plt.show()
    
    
    #from sklearn.model_selection import cross_val_score, KFold

    # df_raw1 = pd.read_excel(r'/Users/rehnan/Library/CloudStorage/OneDrive-Chalmers/Projektet/bridge_data_/csv_2022/raw_data/merged_file_for_regression_final.xlsx')
    # df1 = clean_data(df_raw1)

    # # Assuming df is your DataFrame with the correct columns.
    # # Replace the dummy data with your actual data.
    # df_2 = pd.DataFrame({
    #     'C2': df1['C2'], 
    #     'share': df1['share'], 
    #     'x': df1['x'], 
    #     'ph': df1['ph'], 
    #     'ler': df1['ler'], 
    #     'mat': df1['mat'], 
    #     'map': df1['map'], 
    #     'n': df1['n']
    # })

    # # Set the predictors and dependent variable
    # predictors = ['x', 'ph', 'ler', 'mat', 'map', 'n']
    # dependent_var = 'share'

    # # Initialize Random Forest Regressor
    # rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # # Cross-validation
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # mse_scores = cross_val_score(rf, df_2[predictors], df_2[dependent_var], scoring='neg_mean_squared_error', cv=kf)

    # # Convert negative MSE scores to positive
    # mse_scores = -mse_scores

    # print(f"MSE scores for each fold: {mse_scores}")
    # print(f"Average MSE across all folds: {np.mean(mse_scores)}")
    # print(f"Standard Deviation of MSE across all folds: {np.std(mse_scores)}")

    
randomforestpart()



