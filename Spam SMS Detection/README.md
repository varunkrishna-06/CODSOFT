# Spam SMS Detection

This project focuses on classifying SMS messages as either "spam" or "ham" using machine learning techniques. Here's a detailed description of the steps involved:

#### 1. Load the Dataset
The project starts with loading a dataset of SMS messages from a CSV file into a pandas DataFrame. The dataset contains columns with generic names, which are renamed to `label` for the message classification (spam or ham) and `message` for the text of the SMS.

#### 2. Preprocessing
The preprocessing step involves converting the categorical labels of SMS messages into binary numerical values: 0 for "ham" and 1 for "spam". This transformation is crucial for applying machine learning models.

#### 3. Split the Dataset
The dataset is divided into training and testing sets to allow the model to learn from one portion of the data and be evaluated on a separate, unseen portion. This split ensures a robust assessment of model performance.

#### 4. Feature Extraction using TF-IDF
Text data is transformed into numerical feature vectors using the Term Frequency-Inverse Document Frequency (TF-IDF) method. This approach reflects the importance of each word in the context of the entire dataset and helps in converting text data into a format suitable for machine learning algorithms.

#### 5. Train the Naive Bayes Classifier
A Naive Bayes classifier is trained on the TF-IDF features extracted from the training data. This classifier is effective for text classification tasks due to its ability to handle word frequency data efficiently.

#### 6. Make Predictions
Once trained, the model predicts the labels for the test set. These predictions are then compared to the actual labels to evaluate the model’s accuracy and performance.

#### 7. Evaluate the Model
Model performance is assessed using several metrics:
- **Accuracy**: Measures the proportion of correctly classified messages.
- **Confusion Matrix**: Provides a detailed breakdown of true positives, true negatives, false positives, and false negatives.
- **Classification Report**: Offers precision, recall, and F1-score metrics for both spam and ham classes.

#### 8. Visualization
- **Distribution of Messages by Label**: A count plot visualizes the number of ham and spam messages in the dataset.
- **Confusion Matrix**: A heatmap displays the confusion matrix to show the performance of the model’s predictions.
- **Top Words in Spam and Ham**: Bar plots illustrate the most significant words associated with spam and ham messages based on their log probabilities.
- **Histogram of Message Lengths**: Histograms with KDE curves reveal the distribution of message lengths for both spam and ham messages, highlighting differences between the two classes.

#### 9. Custom Message Prediction
A custom SMS message is tested with the trained model to predict whether it is spam or ham. This demonstrates the practical application of the model to real-world messages.

#### Conclusion
The Spam SMS Detection project successfully implemented a Naive Bayes classifier to distinguish between spam and ham messages with high accuracy. The use of TF-IDF for feature extraction and the evaluation metrics demonstrate the model’s effectiveness. Insights from the analysis, such as the most indicative words and message lengths, enhance understanding of spam characteristics and can improve spam detection systems further. Future work could involve exploring more sophisticated models and integrating additional features to refine performance.
