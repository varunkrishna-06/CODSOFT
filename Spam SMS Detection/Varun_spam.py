import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the Dataset
data = pd.read_csv(r"C:\Users\praji\OneDrive\ドキュメント\Project Varun\Spam SMS Detection\spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Step 2: Preprocessing
# Convert the labels to binary values
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Step 3: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.20, random_state=42)

# Step 4: Feature Extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 5: Train the Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test_tfidf)

# Step 7: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Visualization

# 1. Distribution of Messages by Label
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=data)
plt.title('Distribution of Messages by Label')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Ham', 'Spam'])
plt.show()

# 2. Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 3. Top Words in Spam and Ham
def get_top_n_words(feature_log_prob, feature_names, n=20):
    top_n_indices = feature_log_prob.argsort()[-n:]
    top_n_words = [feature_names[i] for i in top_n_indices]
    return top_n_words, feature_log_prob[top_n_indices]

feature_names = tfidf_vectorizer.get_feature_names_out()

# Get log probabilities for ham and spam
ham_log_prob = model.feature_log_prob_[0]
spam_log_prob = model.feature_log_prob_[1]

# Get top words for ham
ham_top_words, _ = get_top_n_words(ham_log_prob, feature_names)

# Get top words for spam
spam_top_words, _ = get_top_n_words(spam_log_prob, feature_names)

# Plot top words
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.barh(ham_top_words[::-1], ham_log_prob[ham_log_prob.argsort()[-20:]][::-1], color='blue')
plt.title('Top Words in Ham Messages')
plt.xlabel('Log Probability')
plt.gca().invert_yaxis()

plt.subplot(1, 2, 2)
plt.barh(spam_top_words[::-1], spam_log_prob[spam_log_prob.argsort()[-20:]][::-1], color='red')
plt.title('Top Words in Spam Messages')
plt.xlabel('Log Probability')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# 4. Histogram of Message Lengths
data['message_length'] = data['message'].apply(len)
plt.figure(figsize=(8, 6))
sns.histplot(data[data['label'] == 0]['message_length'], color='blue', label='Ham', kde=True)
sns.histplot(data[data['label'] == 1]['message_length'], color='red', label='Spam', kde=True)
plt.title('Histogram of Message Lengths')
plt.xlabel('Message Length')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Optionally, you can test with a custom message
sample_message = ["Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/123456 to claim now."]
sample_message_tfidf = tfidf_vectorizer.transform(sample_message)
prediction = model.predict(sample_message_tfidf)
print("Prediction for sample message:", "Spam" if prediction[0] else "Ham")
