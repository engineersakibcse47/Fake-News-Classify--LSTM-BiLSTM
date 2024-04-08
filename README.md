# Developed a deep learning technique to identify when an article might be fake news.

##### Dataset Description

https://www.kaggle.com/competitions/fake-news/data

`train.csv`: A full training dataset with the following attributes:

`id: unique id for a news article`
`title: the title of a news article`
`author: author of the news article`
`text: the text of the article; could be incomplete`
`label: a label that marks the article as potentially unreliable`
`1: unreliable`
`0: reliable`

`test.csv`: A testing training dataset with all the same attributes at train.csv without the label.

`submit.csv`: A sample submission.


### Introduction

This project explored the use of Long Short-Term Memory (LSTM) networks with Word2Vec embeddings to identify when an article might be fake news(text classification). The goal was to develop a model that could effectively categorize text data into two predefined classes.

### Methodology

#### Data Preprocessing and Feature Engineering:
 
Text data was cleaned to remove noise and irrelevant information.
Stop words were eliminated.
Words were lemmatized to capture their base form.
Text was tokenized and padded to ensure consistent sequence length.

#### Word Embeddings:

Word2Vec was used to generate numerical representations (embeddings) for each word in the vocabulary.
These embeddings capture semantic relationships between words.

#### Model Architecture:

<img width="400" alt="image" src="https://github.com/engineersakibcse47/Fake-News-Classify--LSTM-BiLSTM/assets/108215990/b8b46084-0538-4bdc-b0e3-920abd9fc858">


A Bidirectional LSTM network was employed for text classification.
Bidirectional LSTMs process text in both directions, capturing long-range dependencies.
L1 and L2 regularization were applied to prevent overfitting.
Dropout was used to further prevent overfitting by randomly dropping neurons during training.

#### Training and Evaluation:

The data was divided into training and testing/validation sets.
The model was trained on the training set and evaluated on the unseen testing set.
Early stopping used to prevent overfitting by halting training if validation loss ceased to improve.
Model performance was measured using accuracy, precision, recall, and F1-score.

### Results

<img width="400" alt="image" src="https://github.com/engineersakibcse47/Fake-News-Classify--LSTM-BiLSTM/assets/108215990/25718617-8b46-470b-983e-44e5cd17edec">

<img width="400" alt="image" src="https://github.com/engineersakibcse47/Fake-News-Classify--LSTM-BiLSTM/assets/108215990/90fd9229-c351-4956-bb9d-f86f1e3cb52a">

The final model achieved an accuracy of `97%` on the testing set.
We can see, classification report showed high precision, recall, and F1-score for both classes, indicating a well-balanced and effective model.

### Discussion

The project successfully demonstrated the effectiveness of LSTM networks with Word2Vec embeddings for text classification. The high accuracy and balanced performance across classes suggest the model can accurately distinguish between the two categories.

5. Conclusion

This project showcased the potential of LSTMs and Word2Vec for text classification tasks. The developed model achieved a high level of accuracy and demonstrated a good balance between precision and recall.

6. Future Work

Experiment with different hyperparameters to potentially improve model performance.
Explore alternative text classification models (e.g., GRU,convolutional neural networks).
Apply the model to a new text classification problem.
