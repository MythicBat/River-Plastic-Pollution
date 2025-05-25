import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data/cleaned_river_plastic_data.csv")

# Risk Level categorization
def categorize_risk(score):
    if score < 1:
        return 'Low'
    elif score < 3:
        return 'Medium'
    else:
        return 'High'

df['Risk_Level'] = df['Risk_Score_Change'].apply(categorize_risk)

# Encode target Labels
label_encoder = LabelEncoder()
df['Risk_Level_Label'] = label_encoder.fit_transform(df['Risk_Level'])  # Low = 1, Medium = 2, High = 0 (varies)

# Features and Labels
features = [
    'Population_2015',
    'Urbanization_2015_pct',
    'Policy_Strength_2015',
    'Waste_Collection_Rate_2015',
    'Plastic_to_River_2015_tons'
]

X = df[features]
y = df['Risk_Level_Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n{name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
