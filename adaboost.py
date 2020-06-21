import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
# Import train_test_split function
from sklearn.model_selection import train_test_split
# load required classifer
from sklearn.ensemble import AdaBoostClassifier
# import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score, classification_report

network = pd.read_csv('sdn_datasets/train/train.1000.csv')
network2 = pd.read_csv('sdn_datasets/test/test.10000.csv')
network.head()
network.info()
network2.head()
network2.info()
X = network[['dur', 'stddev', 'min', 'mean', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'max', 'sum']]
X2 = network2[['dur', 'stddev', 'min', 'mean', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'max', 'sum']]

y = network['label']
y2 = network2['label']


le = LabelEncoder()

y = le.fit_transform(y)
y2 = le.fit_transform(y2)


# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=5, learning_rate=0.3, random_state=0)

# Train Adaboost Classifer
model = abc.fit(X, y)

# Predict the response for test dataset
y_pred = model.predict(X2)
# calculate and print model accuracy
print("AdaBoost Classifier Model Accuracy:", accuracy_score(y2, y_pred))

print(classification_report(y2, y_pred, labels=[0, 1, 2, 3, 4, 5]))
