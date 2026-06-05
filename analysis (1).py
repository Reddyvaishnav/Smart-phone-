import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("smartphone_data.csv")

# Convert Gender to numbers
df["Gender"] = df["Gender"].map({
    "Male": 1,
    "Female": 0
})

# Features and Target
X = df[[
    "Gender",
    "ScreenTime",
    "SocialMediaHours",
    "SleepHours"
]]

y = df["AddictionRisk"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n========== MODEL PERFORMANCE ==========\n")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Feature Importance
importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

print("\n========== FEATURE IMPORTANCE ==========\n")
print(importance_df)

# Feature Importance Graph
plt.figure(figsize=(8,5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df
)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")

# SHAP Explainability
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

try:
    shap.summary_plot(
        shap_values,
        X_test,
        show=False
    )
except:
    shap.summary_plot(
        shap_values[1],
        X_test,
        show=False
    )

plt.savefig("shap_plot.png")

# Bias Analysis
male_risk = df[df["Gender"] == 1][
    "AddictionRisk"
].mean()

female_risk = df[df["Gender"] == 0][
    "AddictionRisk"
].mean()

print("\n========== BIAS ANALYSIS ==========\n")

print("Male High Risk Rate:",
      round(male_risk * 100, 2), "%")

print("Female High Risk Rate:",
      round(female_risk * 100, 2), "%")

# Bias Chart
plt.figure(figsize=(6,5))

plt.bar(
    ["Male", "Female"],
    [male_risk * 100,
     female_risk * 100]
)

plt.title("Addiction Risk by Gender")

plt.ylabel("Percentage")

plt.savefig("bias_analysis.png")

# User Prediction
print("\n========== SMARTPHONE ADDICTION PREDICTION ==========\n")

gender = input("Enter Gender (Male/Female): ")

screen = float(
    input("Enter Daily Screen Time (hours): ")
)

social = float(
    input("Enter Social Media Hours: ")
)

sleep = float(
    input("Enter Sleep Hours: ")
)

if gender.lower() == "male":
    gender_value = 1
else:
    gender_value = 0

prediction = model.predict([[
    gender_value,
    screen,
    social,
    sleep
]])

probability = model.predict_proba([[
    gender_value,
    screen,
    social,
    sleep
]])[0]

if prediction[0] == 1:

    print("\n========== ADDICTION REPORT ==========\n")

    print("Risk Level: HIGH 🔴")

    print("Confidence:",
          round(max(probability) * 100, 2),
          "%")

else:

    print("\n========== ADDICTION REPORT ==========\n")

    print("Risk Level: LOW 🟢")

    print("Confidence:",
          round(max(probability) * 100, 2),
          "%")

print("\n========== MITIGATION RECOMMENDATIONS ==========\n")

print("1. Reduce daily screen time")
print("2. Limit social media usage")
print("3. Sleep at least 7–8 hours")
print("4. Use digital wellbeing apps")
print("5. Take regular breaks from screens")