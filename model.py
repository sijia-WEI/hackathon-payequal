import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
import numpy as np


# Load your dataset
df_loc = 'salaries.csv'
df = pd.read_csv(df_loc)  # Replace with your CSV file path

# Select features and target
X = df[['experience_level', 'employment_type', 'job_title',
        'remote_ratio', 'company_location', 'company_size']]

y = df['salary_in_usd']

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import train_test_split

# Custom wrapper for log-transforming target
class LogTargetRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_model):
        self.base_model = base_model
    
    def fit(self, X, y):
        # Log-transform the target variable
        self.y_log = np.log(y)
        self.base_model.fit(X, self.y_log)
        return self
    
    def predict(self, X):
        # Make predictions and revert log transformation
        log_predictions = self.base_model.predict(X)
        return np.exp(log_predictions)  # Convert predictions back to original scale
    
    def score(self, X, y):
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    
experience_order = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
X['experience_level'] = X['experience_level'].map(experience_order)

# Define categorical columns
categorical_features = ['employment_type', 'job_title', 'company_location', 'company_size']

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')  # remote_ratio will pass through as numeric

# Model
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LogTargetRegressor(
        RandomForestRegressor(n_estimators=100, random_state=39)))
])

rf_model.fit(X, y)

# Custom input
new_input = pd.DataFrame([{
    'experience_level': 'SE',
    'employment_type': 'FT',
    'job_title': 'Data Scientist',
    'remote_ratio': 100,
    'company_location': 'US',
    'company_size': 'L'
}])

new_input['experience_level'] = new_input['experience_level'].map(experience_order)

predicted_salary = rf_model.predict(new_input)