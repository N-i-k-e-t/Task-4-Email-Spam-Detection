# Spam Detection Model

## Internship Details

* **Company:** The Black Pearls
* **Internship Period:** June 15, 2024 - July 15, 2024 

## Introduction

This project aims to build a spam detection model using machine learning techniques. The model is trained on a dataset of SMS messages, learning to classify new messages as either spam or ham (legitimate).

**Business Context:** Spam detection is crucial for businesses to protect users from unwanted and potentially malicious communication. This project demonstrates the ability to apply data science techniques to address this common issue, showcasing skills in natural language processing, model building, and evaluation.

## Dataset and Features

* **Data Source:**  The project utilizes the "SMS Spam Collection" dataset, commonly used for spam detection tasks.
* **Features:** 
    * **text:**  The content of each SMS message (raw text).
    * **label:**  The classification of the message ('spam' or 'ham').
* **Data Preprocessing:**
    * **Text Cleaning:** Lowercasing, punctuation removal.
    * **Tokenization:** Splitting text into individual words.
    * **Stop Word Removal:**  Removing common words like "the", "a", "is", etc.
    * **Stemming:**  Reducing words to their root form (e.g., "running" to "run").

## Methodology

* **Approach:** Supervised machine learning was used for this classification task.
* **Algorithms:**  
    * **Multinomial Naive Bayes:** A probabilistic algorithm well-suited for text classification.
* **Model Selection:** 
    * **Grid Search with Cross-Validation:** Used to find the optimal hyperparameters for the Multinomial Naive Bayes model (specifically, tuning the `alpha` parameter) and ensure robust performance estimation. The best performing model was `Pipeline(steps=[('tfidf', TfidfVectorizer(max_features=5000)), ('model', MultinomialNB(alpha=0.1))])`. 

## Results and Evaluation

* **Key Findings:**  The model effectively learned to differentiate spam from ham based on word patterns and frequencies.
* **Performance Metrics:**
    * **Accuracy:**  The model achieved an accuracy of 98.39% on the test set, indicating strong predictive capability.
    * **Confusion Matrix:**  [Include a screenshot or link to the confusion matrix visualization]
    * **Classification Report:**  [Include a screenshot or the text output of the classification report]
* **Visualizations:**
    * **Confusion Matrix Heatmap:**  [Screenshot or link]
    * **Word Clouds:**  [Screenshot or link] 

## Conclusion

* **Summary:**  The project successfully developed a spam detection model, achieving high accuracy (98.39%) in classifying SMS messages. 
* **Business Implications:** This model could be integrated into email or messaging platforms to automatically flag or filter spam, enhancing user experience and security.
* **Future Work:** 
    * Experiment with more sophisticated algorithms (e.g., Support Vector Machines, Recurrent Neural Networks) for potentially improved performance.
    * Explore techniques like sentiment analysis to identify different types of spam (e.g., phishing attempts vs. promotional emails).

## Technical Information

* **Installation:**
    * The project uses Python and requires the following libraries: pandas, scikit-learn, nltk, wordcloud, matplotlib, seaborn.
    * A `requirements.txt` file is provided with the project, listing the specific versions of the dependencies used. 
* **How to Run:**
    1. Ensure all necessary libraries are installed by running `pip install -r requirements.txt`.
    2. Download the SMS Spam Collection dataset and update the file path in the code.
    3. Run the Python script to train and evaluate the model. 

## Contact Information

* **LinkedIn:**  [https://www.linkedin.com/in/niket-patil-/](https://www.linkedin.com/in/niket-patil-/ "https://www.linkedin.com/in/niket-patil-/")
* **Email:** niketpatil1624@gmail.com
