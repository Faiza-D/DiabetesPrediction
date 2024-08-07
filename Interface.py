import tkinter as tk
from tkinter import messagebox
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load the dataset and preprocess as before
df = pd.read_csv('C:/PythonProject/Diabetes-Prediction-Model/diabetes.csv')
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[columns_with_zeros] = df[columns_with_zeros].replace(0, np.nan)
for column in columns_with_zeros:
    df.loc[:, column] = df[column].fillna(df[column].median())
X = df.drop(columns='Outcome', axis=1)
Y = df['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression()
model.fit(X_scaled, Y)

# Create the Tkinter window
root = tk.Tk()
root.title("Diabetes Prediction")

# Set a larger font size and background color
font = ('Arial', 14)
bg_color = '#f0f8ff'  # AliceBlue background color
label_color = '#4682b4'  # SteelBlue color for labels
button_color = '#ff6347'  # Tomato color for buttons
result_color = '#32cd32'  # LimeGreen color for result text

root.configure(bg=bg_color)

# Function to make a prediction
def predict():
    try:
        features = [float(entry.get()) for entry in entries]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        result_text = "Diabetic" if prediction == 1 else "Not Diabetic"
        proba_text = f"Probabilities - Not Diabetic: {proba[0]:.2f}, Diabetic: {proba[1]:.2f}"
        result_label.config(text=f"{result_text}\n{proba_text}", fg=result_color)
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

# Create input fields for each feature
labels = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
entries = []

for i, label in enumerate(labels):
    tk.Label(root, text=label, font=font, bg=bg_color, fg=label_color).grid(row=i, column=0, padx=10, pady=5, sticky='w')
    entry = tk.Entry(root, font=font)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

# Add a button to trigger the prediction
tk.Button(root, text="Predict", command=predict, font=font, bg=button_color, fg='white').grid(row=len(labels), columnspan=2, pady=10)

# Add a label to display the result
result_label = tk.Label(root, text="", font=('Arial', 16), bg=bg_color)
result_label.grid(row=len(labels)+1, columnspan=2, pady=10)

# Run the Tkinter event loop
root.mainloop()
