import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import randint

train_df = pd.read_csv("C:/Users/janak/Downloads/archive/fraudTrain.csv")
test_df = pd.read_csv("C:/Users/janak/Downloads/archive/fraudTest.csv")

plt.figure(figsize=(6, 4))
sns.countplot(x='is_fraud', data=train_df)
plt.title('Distribution of Fraudulent vs Non-Fraudulent Transactions')
plt.show()

numeric_features = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
train_df[numeric_features].hist(bins=30, figsize=(15, 10), layout=(3, 3))
plt.suptitle('Distribution of Numeric Features')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(train_df[numeric_features].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap of Numeric Features')
plt.show()

categorical_features = ['merchant', 'category', 'gender']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(
    train_df.drop(columns=['is_fraud']), 
    train_df['is_fraud'], 
    test_size=0.2, 
    random_state=42, 
    stratify=train_df['is_fraud'])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

rus = RandomUnderSampler(random_state=42)
X_train_downsampled, y_train_downsampled = rus.fit_resample(X_train_processed, y_train)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_downsampled, y_train_downsampled)

rf_classifier = RandomForestClassifier(random_state=42)

param_dist = {
    'n_estimators': randint(100, 200),
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': randint(10, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4),
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_dist, n_iter=50, cv=3, n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train_resampled, y_train_resampled)

best_rf_classifier = random_search.best_estimator_

feature_importances = best_rf_classifier.feature_importances_
features = numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance in RandomForest Classifier')
plt.show()

y_pred = best_rf_classifier.predict(X_test_processed)

print("Test Set Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nTest Set Classification Report:")
print(classification_report(y_test, y_pred))
print("\nROC AUC Score on Test Set:", roc_auc_score(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

y_pred_proba = best_rf_classifier.predict_proba(X_test_processed)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='RandomForest (AUC = {:.2f})'.format(roc_auc_score(y_test, y_pred_proba)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='RandomForest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

print("\nBest Parameters found by RandomizedSearchCV:")
print(random_search.best_params_)
