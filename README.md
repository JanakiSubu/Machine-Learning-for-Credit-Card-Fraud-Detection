# Credit Card Fraud Detection Using Machine Learning

## Overview

This project focuses on developing a machine learning model to detect fraudulent credit card transactions. Detecting fraud is crucial for financial institutions to protect their customers and prevent financial losses. Leveraging advanced machine learning techniques, this project aims to build a robust model that accurately identifies fraudulent transactions while minimizing false positives.

## Requirements

- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, scipy

## Installation

To install the required libraries, run the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn scipy
```

## Dataset

The dataset used in this project consists of transactions labeled as fraudulent or non-fraudulent. It includes features such as transaction amount, location details, merchant information, and more. The dataset is split into training and test sets for model evaluation.

## Exploratory Data Analysis (EDA)

- Visualized the distribution of fraudulent vs non-fraudulent transactions.
- Analyzed the distribution of numeric features and their correlations using histograms and heatmaps.

## Preprocessing

- Standardized numeric features and encoded categorical features using appropriate transformers from scikit-learn's preprocessing module.
- Handled class imbalance using Synthetic Minority Over-sampling Technique (SMOTE) and undersampling of the majority class.

## Model Building

- Utilized Random Forest Classifier due to its ability to handle complex datasets and mitigate overfitting.
- Employed RandomizedSearchCV for hyperparameter tuning to optimize model performance.

## Evaluation

- Evaluated the model using metrics such as confusion matrix, classification report, ROC AUC score, and precision-recall curve.
- Visualized model performance metrics including ROC curve and precision-recall curve for comprehensive analysis.

## Results

- Achieved an accuracy of 98% with a recall rate of 78% for detecting fraudulent transactions.
- The model's ROC AUC score of 0.88 indicates its ability to discriminate between fraudulent and non-fraudulent transactions effectively.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Install dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn scipy
   ```

3. Run the Jupyter notebook or Python script to train the model and evaluate its performance:

   ```bash
   jupyter notebook fraud_detection.ipynb
   ```

## Conclusion

This project demonstrates the application of machine learning in enhancing credit card fraud detection systems. By leveraging data preprocessing, advanced modeling techniques, and comprehensive evaluation, the developed model provides a reliable solution to mitigate fraudulent activities in financial transactions.

If you're interested in data science, fraud detection, or have any feedback or questions about the project, I'd love to connect with you on LinkedIn. Let's discuss how data science can tackle real-world challenges together!

Connect with me here:  https://www.linkedin.com/in/janaki-subramani-681420177/

Looking forward to connecting and sharing insights!
