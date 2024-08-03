# Sentiment Analysis on Amazon Fine Food Reviews

## Introduction

This project aims to perform sentiment analysis on the Amazon Fine Food Reviews dataset. Sentiment analysis is a common task in natural language processing (NLP) that involves determining whether a piece of text (such as a review) expresses a positive, negative, or neutral sentiment. In this project, we use an LSTM neural network to classify reviews as positive or negative.

## Dataset

The dataset used in this project is the Amazon Fine Food Reviews dataset, which consists of around 500,000 reviews of fine foods from Amazon. The dataset includes the following features:
- `Id`: Unique identifier for each review.
- `ProductId`: Unique identifier for each product.
- `UserId`: Unique identifier for each user.
- `ProfileName`: Profile name of the user.
- `HelpfulnessNumerator`: Number of users who found the review helpful.
- `HelpfulnessDenominator`: Number of users who indicated whether they found the review helpful.
- `Score`: Rating of the product (1 to 5 stars).
- `Time`: Timestamp for the review.
- `Summary`: Brief summary of the review.
- `Text`: Full text of the review.

For the purpose of sentiment analysis, we convert the `Score` into binary labels where ratings of 4 and 5 are considered positive (1) and ratings of 1, 2, and 3 are considered negative (0).

## Project Workflow

1. **Data Preparation and Cleaning:**
   - Load the dataset and perform initial cleaning.
   - Convert the `Score` into binary labels for sentiment analysis.
   - Clean the text data by removing URLs, HTML tags, punctuation, newline characters, and words containing digits.
   - Tokenize the text and remove stopwords.
   - Stem the remaining words for normalization.

2. **Tokenization and Padding:**
   - Use the `Tokenizer` class from TensorFlow to convert the text data into sequences of integer indices.
   - Apply padding to ensure all sequences have the same length, which is necessary for training LSTM models.

3. **Model Architecture:**
   - Define a Sequential model with the following layers:
     - An Embedding layer to convert word indices to dense vectors of fixed size.
     - An LSTM layer with 50 units to capture temporal dependencies in the data.
     - A Dense output layer with a softmax activation function for binary classification.
   - Compile the model using categorical cross-entropy loss and the Adam optimizer.

4. **Model Training:**
   - Train the model on the preprocessed and tokenized text data.

5. **Model Evaluation:**
   - Evaluate the model using accuracy and F1-score.
   - Visualize the model's performance using a confusion matrix.

## Results

The model achieved an accuracy of `92%`. The confusion matrix provided insights into the classification performance, showing how well the model distinguished between positive and negative reviews.

## Conclusion

In this project, we successfully performed sentiment analysis on the Amazon Fine Food Reviews dataset using an LSTM neural network. The key steps involved data preparation, tokenization, model training, and evaluation. The results indicate that the LSTM model is effective for this sentiment analysis task.

## Future Work

Future improvements could include:
- Exploring more advanced models such as BERT or other deep learning techniques to potentially enhance performance.
- Performing hyperparameter tuning to optimize the LSTM model further.
- Experimenting with different text preprocessing techniques and feature extraction methods.


