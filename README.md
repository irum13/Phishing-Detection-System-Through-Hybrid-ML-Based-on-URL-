# 🚀 Phishing Detection System through Machine Learning Based on URL's

## 📌 Abstract
Phishing attacks are a major cybersecurity threat, primarily using email distortion and fake websites. Despite multiple studies on phishing prevention, there is no fully effective method. This project leverages **machine learning** to detect phishing URLs using a dataset of **11,000+ websites**. Various models like **Decision Trees, Random Forest, SVM, Gradient Boosting, and a hybrid LSD model** (combining Logistic Regression, SVM, and Decision Trees) are implemented to enhance detection accuracy.

## 🔍 Existing System
Phishing detection systems fall into two main categories:
1. **List-Based Methods** (Blacklists & Whitelists)
2. **Machine Learning-Based Methods**

### ❌ Disadvantages of Existing Systems
- **URLs were collected from a single anti-spam provider**, limiting diversity.
- **Blacklists fail to provide zero-hour protection**, detecting less than 20% of phishing attempts immediately.
- **Limited to email-based URLs**, missing other attack vectors.
- **Lack of hybrid machine learning models**, leading to reduced detection performance.

## ✅ Proposed System
Our model uses a **phishing URL dataset** from a reputable repository with features extracted from **11,000+ websites**. 

### 🔹 Advantages of Proposed System
- **Higher prediction accuracy**
- **Hybrid model (LSD)** combining **Logistic Regression, SVM, and Decision Trees** for better phishing detection.
- **Canopy feature selection** and **Grid Search Hyperparameter Optimization** for enhanced model performance.

## 🔧 Functional Requirements
1. **Data Collection**
2. **Data Preprocessing**
3. **Training and Testing**
4. **Modeling**
5. **Prediction**

## 🏗️ System Architecture
### 📌 Implementation Modules
- **Data Exploration**: Load dataset into the system.
- **Processing**: Read and preprocess data.
- **Splitting Data**: Divide data into training and testing sets.
- **Model Generation**: Train machine learning models.
- **User Signup & Login**: Registration and authentication.
- **User Input**: Accept URLs for phishing detection.
- **Prediction**: Display the final phishing URL detection result.

## 🧠 Algorithms Used
| Algorithm  | Description |
|------------|------------|
| **Linear Regression (LR)** | Predicts dependent variables using independent variables. |
| **Random Forest** | Combines multiple decision trees for better accuracy. |
| **Decision Tree** | A tree-based classification model for decision making. |
| **Support Vector Machine (SVM)** | Finds optimal data classification boundaries. |
| **Naive Bayes** | Probabilistic model based on Bayes' theorem. |
| **Gradient Boosting** | Uses boosting techniques to improve performance. |
| **Hybrid LSD** | Combination of LR, SVM, and Decision Trees using soft and hard voting. |
| **Stacking Classifier (RF + MLP with LightGBM)** | An ensemble method stacking multiple models for better predictions. |

## 📌 Conclusion
Phishing URLs act as legitimate links, tricking users into exposing sensitive information. This study implements a **machine learning-based phishing detection system** using **32 URL attributes** from **11,000+ URLs**. 

Our model employs advanced techniques like **canopy feature selection, cross-validation, and Grid Search Hyperparameter Optimization** for maximum accuracy. The proposed hybrid LSD model outperforms traditional methods, offering better protection against phishing attacks.

### 🚀 Future Work
- Integrating **list-based methods** with **machine learning** to improve phishing detection.
- Expanding detection to **social media and messaging platforms** beyond email-based phishing attacks.

## 🛠️ Installation & Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/irum13/Phishing-Detection-System-Through-Hybrid-ML-Based-on-URL-.git
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the model:
   ```sh
   python app.py
   ```
## 📖 Phishing Detection System Notebook

You can view the full Jupyter Notebook on **Nbviewer** here:  
🔗 [View Notebook](https://nbviewer.org/github/irum13/Phishing-Detection-System-Through-Hybrid-ML-Based-on-URL-/blob/main/org.ipynb)

If you want to run this notebook online, open it in **Google Colab**:  
🚀 [Run on Colab](https://colab.research.google.com/github/irum13/Phishing-Detection-System-Through-Hybrid-ML-Based-on-URL-/blob/main/org.ipynb)

## 📸 Application Screenshots

### 🏠 Home Page
![Home Page](home_page.png)

### 📝 User Signup
![User Signup](sign_up_page.png)

### 🔐 User Login
![User Login](login_page.png)

### 🔍 URL Search Page
![URL Search Page](url_search_page.png)

### 📊 URL Result Page 1
![Result Page 1](url_result_page_1.png)

### 📊 URL Result Page 2
![Result Page 2](url_result_page_2.png)

---
📢 *If you like this project, don't forget to star ⭐ the repo!*
