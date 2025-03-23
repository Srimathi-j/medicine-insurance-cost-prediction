Medicinal insurance cost prediction Implementation Details:
  Methodology & Approach
  The model development process involves:
1.	Data Preprocessing:
       Convert categorical variables (sex, smoker, region) into numeric values.
       Normalize numerical features (age, bmi, children) using StandardScaler.
       Handle missing values (if any) and check for outliers.
2.	Exploratory Data Analysis (EDA):
            Correlation heatmap to identify key predictors.
            Visualization of smoking impact on insurance charges.
            Distribution plots of insurance costs.
3.	Feature Selection:
            Identify the most influential features using correlation analysis.
            smoker, bmi, and age are the most significant predictors.
4.	Model Selection & Training
5.	Train three models: 
             Linear Regression – Baseline model.
             Random Forest – Captures complex relationships.
             XGBoost – Optimized for better performance.
6.	Hyperparameter tuning for XGBoost to improve accuracy.
7.	Performance Evaluation:
           Compare models using R² score, Mean Absolute Error (MAE), Mean Squared Error (MSE), and RMSE.
8. Visualize actual vs predicted insurance costs.
