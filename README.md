Project Overview

This project uses Machine Learning (Random Forest Classifier) to predict whether a user is at High Risk or Low Risk of smartphone addiction based on behavioral factors such as:

Gender
Daily Screen Time
Social Media Usage Hours
Sleep Hours

The project also incorporates Explainable AI (XAI) using SHAP (SHapley Additive Explanations) to interpret model predictions and includes a basic bias analysis to compare addiction risk across genders.


🎯 Objectives

Predict smartphone addiction risk using behavioral data.
Identify the most influential factors contributing to addiction.
Improve transparency through explainable AI techniques.
Analyze potential bias in predictions across different genders.
Provide recommendations for reducing smartphone addiction risk.

🛠️ Technologies Used

Python
Pandas
Scikit-Learn
Matplotlib
Seaborn
SHAP (Explainable AI)

📂 Dataset Features

Feature	Description
Gender	Male or Female
ScreenTime	Daily smartphone usage in hours
SocialMediaHours	Time spent on social media daily
SleepHours	Average sleep duration
AddictionRisk	Target variable (0 = Low Risk, 1 = High Risk)
