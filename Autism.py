import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r'C:\Users\adlin\OneDrive\Desktop\machine learning projects\autism-prediction-using-machine-learning.csv')

# Check the structure of your data (optional)
print(df.head())
df['Autism_Risk'] = (df[['Speech_Delay', 'Social_Interaction', 'Imaginative_Play', 
                          'Repetitive_Behaviors', 'Sensory_Sensitivity', 'Eye_Contact', 
                          'Parental_Concern']].sum(axis=1) > 3).astype(int)

# Features and target variable
X = df[['Age', 'Speech_Delay', 'Social_Interaction', 'Imaginative_Play', 
        'Repetitive_Behaviors', 'Sensory_Sensitivity', 'Eye_Contact', 
        'Parental_Concern']]
y = df['Autism_Risk']

# Split the data into training and testing sets with a fixed random seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the RandomForestClassifier with balanced class weights and random_state
model = RandomForestClassifier(class_weight='balanced', random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics with zero_division to avoid undefined metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

# Print the results
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Plot Confusion Matrix (without Seaborn)
fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.matshow(cm, cmap='Blues')
fig.colorbar(cax)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0, 1], ['Not at risk', 'At risk'])
plt.yticks([0, 1], ['Not at risk', 'At risk'])
plt.show()

# Precision-Recall Curve (without Seaborn)
precision_vals, recall_vals, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])

plt.figure(figsize=(6, 6))
plt.plot(recall_vals, precision_vals, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.show()
