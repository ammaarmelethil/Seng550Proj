# 🎥 Sentiment Analysis of YouTube Comments 📊

## Overview
This project focuses on **sentiment analysis** of YouTube comments, classifying them into three categories:
- **Negative** (0)
- **Neutral** (1)
- **Positive** (2)

Using a dataset of pre-labeled comments from five diverse YouTube videos, we built a scalable **data engineering pipeline** to preprocess, analyze, and classify sentiments. Our findings provide meaningful insights into user engagement and comment behavior while demonstrating the utility of combining **PySpark** with **machine learning** techniques.

---

## Project Highlights 🚀
- **Dataset**: Comments sourced from five YouTube videos on topics ranging from Logan Paul’s apology video to Taylor Swift’s music video and Donald Trump political commentary.
- **Pipeline**: 
  - Text preprocessing: stopword removal, lemmatization, and tokenization.
  - Feature engineering: CountVectorizer and TF-IDF for robust text representation.
  - Model training: Logistic Regression with class weighting to handle imbalances.
- **Accuracy**: Achieved **90.57% cross-validated test accuracy**.
- **Insights**:
  - Negative comments tend to be longer on average than positive ones.
  - Domain-specific terms like "paper" and "printer" were prominent due to unique video content but removed during preprocessing to enhance model accuracy.

---

## Tools and Frameworks 🛠️
- **PySpark**: For scalable distributed data processing and machine learning.
- **Python Libraries**:
  - **NLTK**: For stopword removal and lemmatization.
  - **Matplotlib**: For data visualization.
  - **Pandas**: For lightweight data analysis and exploration.
- **Machine Learning Framework**: PySpark MLlib for model training and evaluation.
- **IDE**: Jupyter Notebook for development and exploratory analysis.

---

## Key Findings 📈
- **Precision and Recall**:
  - Negative: Precision: 95.67%, Recall: 88.16%, F1-Score: 91.76%
  - Neutral: Precision: 84.19%, Recall: 97.68%, F1-Score: 90.43%
  - Positive: Precision: 98.37%, Recall: 82.30%, F1-Score: 89.62%
- **Confusion Matrix** (See [Appendix 5](#)): Highlighted occasional misclassifications between Positive and Neutral comments due to overlapping language patterns.

---

## Methodology ⚙️
1. **Data Ingestion**: Combined datasets into a single PySpark DataFrame after cleaning for missing values.
2. **Exploratory Data Analysis (EDA)**: Analyzed class distribution, text length patterns, and token frequency distributions.
3. **Text Preprocessing**:
   - Removed generic and domain-specific stopwords.
   - Applied lemmatization to normalize text.
4. **Feature Engineering**:
   - Used **CountVectorizer** for token-to-feature conversion.
   - Applied **TF-IDF** to emphasize meaningful tokens.
5. **Model Training**: Built a **Logistic Regression model** with class weighting and optimized using k-fold cross-validation.

---

## Future Improvements 🔮
- Incorporate **deep learning models** like LSTMs or Transformers to better capture sentiment nuances.
- Expand the dataset to include a wider variety of topics for better generalizability.
- Experiment with advanced preprocessing techniques to further refine the feature space.

---

## Conclusion 📝
This project showcases the power of scalable data engineering and machine learning for sentiment analysis. The findings are not only relevant to researchers but also provide actionable insights for content creators and platform moderators to better understand audience engagement.

---

## How to Use 💻
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/sentiment-analysis-youtube.git
