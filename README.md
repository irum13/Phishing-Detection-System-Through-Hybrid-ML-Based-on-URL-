# Phishing-Detection-System-Through-Hybrid-ML-Based-on-URL-
**Overview
**
Phishing attacks are among the most prevalent cybersecurity threats, where attackers create fraudulent websites to deceive users into providing sensitive information. This project aims to detect phishing URLs using various machine learning techniques. A dataset containing phishing and legitimate URLs is analyzed, preprocessed, and used to train several models, including decision trees, support vector machines, random forests, and a hybrid LSD model.

Features

Phishing URL Dataset: Collected from over 11,000 websites, containing 11,054 URLs with 32 attributes.

Machine Learning Models: Includes decision trees, linear regression, random forests, naive Bayes, gradient boosting classifiers, K-nearest neighbors, support vector classifiers, and a hybrid LSD model.

Feature Selection & Hyperparameter Tuning: Canopy feature selection, cross-fold validation, and Grid Search Hyperparameter Optimization for enhanced model performance.

Evaluation Metrics: Accuracy, precision, recall, F1-score, and specificity are used to assess performance.

Existing System & Limitations

Existing Approaches:

Traditional methods relied on blacklists and whitelists, which are ineffective for detecting zero-hour phishing attacks.

Prior studies used data from a single anti-spam service provider, limiting generalization.

Machine-learning-based approaches were implemented but lacked hybrid model efficiency.

Limitations:

Blacklists detected fewer than 20% of phishing attacks at zero-hour.

Delayed updates in heuristics-based detection impacted real-time phishing prevention.

No use of hybrid machine learning models, leading to reduced performance.

Proposed System

Our proposed system enhances phishing detection using a hybrid LSD model that combines Decision Trees, Support Vector Machines, and Logistic Regression with both soft and hard voting mechanisms. Key improvements include:

Hybrid Model: Uses logistic regression, SVM, and decision trees for better accuracy.

Feature Selection & Hyperparameter Optimization: Canopy selection with cross-fold validation and grid search.

Higher Accuracy & Efficiency: Outperforms existing methods in phishing URL detection.

System Architecture

Modules Implemented:

Data Collection: Extract phishing and legitimate URLs from Kaggle repository.

Data Preprocessing: Clean, normalize, and transform dataset for training.

Model Training & Testing: Train machine learning models and evaluate performance.

Prediction System: A functional interface where users input URLs to check for phishing risks.

User Authentication: Sign-up and login functionalities.

Algorithms Used:

Linear Regression (LR)

Random Forest (RF)

Decision Tree (DT)

Support Vector Machine (SVM)

Naive Bayes (NB)

Gradient Boosting (GBM)

Hybrid LSD Model (LR + SVC + DT with soft and hard voting)

Stacking Classifier (RF + MLP with LightGBM)

Implementation Details

Dependencies

Ensure you have the following dependencies installed:

pip install numpy pandas scikit-learn matplotlib seaborn lightgbm

Running the Project

Clone the Repository

git clone https://github.com/your-repo-name.git
cd phishing-url-detection

Run the Python Script

python phishing_detection.py

User Input for Prediction
Enter a URL in the system and check whether it is phishing or legitimate.

Evaluation Metrics

The models are evaluated using:

Accuracy: Measures overall correctness.

Precision: Indicates the percentage of correctly identified phishing URLs.

Recall: Measures the detection rate of phishing URLs.

F1-score: Balances precision and recall.

Specificity: Measures true negative rate.

Conclusion

The proposed hybrid LSD model significantly improves phishing URL detection by combining multiple machine learning models. By leveraging canopy feature selection and hyperparameter tuning, our approach enhances accuracy and robustness against phishing attacks. Future work involves integrating list-based approaches with machine learning models for even more effective phishing prevention.

Contribution

Fork the repository and make improvements.

Submit a pull request for review.

Report issues or suggest enhancements in the GitHub Issues section.

