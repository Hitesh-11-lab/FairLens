import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000
gender = np.random.choice(['Male', 'Female'], n)
credit_score = np.random.normal(650, 50, n)
prob_approve = 1 / (1 + np.exp(-(credit_score - 600)/100))
prob_approve[gender == 'Female'] *= 0.7
approved = np.random.binomial(1, prob_approve)

df = pd.DataFrame({'gender': gender, 'credit_score': credit_score, 'approved': approved})
df.to_csv('loans_biased.csv', index=False)
print("✅ Created loans_biased.csv with columns:", df.columns.tolist())