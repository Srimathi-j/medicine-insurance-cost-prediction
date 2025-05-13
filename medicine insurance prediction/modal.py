import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import joblib

# 1. Load dataset
df = pd.read_csv('insurance.csv')

# 2. Rename target column for consistency
df.rename(columns={'charges': 'insurance_cost'}, inplace=True)

# 3. Encode ordinal/binary features
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
df['region'] = df['region'].map({'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3})
df['activity_level'] = df['activity_level'].map({'low': 0, 'moderate': 1, 'high': 2})

# 4. Label-encode free-text categories
encoders = {}
for col in ['family_history', 'current_medication', 'medical_history']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    joblib.dump(le, f'{col}_encoder.pkl')

# 5. Impute numeric missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# 6. Split features/target
features = ['age','sex','bmi','children','smoker','region',
            'sleep_quality','income_level','diet_quality','activity_level',
            'family_history','current_medication','medical_history']
X = df[features]
y = df['insurance_cost']

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 8. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')

# 9. Train models
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_scaled, y_train)
gb = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_train_scaled, y_train)
xg = xgb.XGBRegressor(n_estimators=100, random_state=42).fit(X_train_scaled, y_train)

# 10. Save models
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(gb, 'gb_model.pkl')
joblib.dump(xg, 'xgb_model.pkl')

# 11. Evaluate ensemble
rf_pred  = rf .predict(X_test_scaled)
gb_pred  = gb .predict(X_test_scaled)
xgb_pred = xg .predict(X_test_scaled)
final_pred = (rf_pred + gb_pred + xgb_pred) / 3
rmse = np.sqrt(mean_squared_error(y_test, final_pred))
print(f'Final ensemble RMSE: {rmse:.2f}')
