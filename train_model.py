import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the biased dataset
df = pd.read_csv('loans_biased.csv')

# Define features (X) and target (y)
X = df[['credit_score']]          # You can add more features here
y = df['approved']

# Train a simple model (Random Forest)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# Save the model to a pickle file
with open('biased_loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved as 'biased_loan_model.pkl'")