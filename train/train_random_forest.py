import sys

import pandas as pd
import utils.functions as functions
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

try:
    DATASET_VERSION = sys.argv[1]
except IndexError:
    DATASET_VERSION = "vx"

print(f"dataset {DATASET_VERSION}")

train_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/train.csv", index_col=0)
test_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/test.csv", index_col=0)
valid_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/valid.csv", index_col=0)

train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)


# Convert text data into TF-IDF features
tfidf_vectorizer = TfidfVectorizer()

x_train = tfidf_vectorizer.fit_transform(train_df["features"].values)
x_test = tfidf_vectorizer.transform(test_df["features"].values)

y_train = train_df["labels"].values
y_test = test_df["labels"].values

decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
y_pred_dt = decision_tree.predict(x_test)

functions.compute_metrics(y_test, y_pred_dt)

random_forest = RandomForestClassifier(n_estimators=50, random_state=42)
random_forest.fit(x_train, y_train)
y_pred_rf = random_forest.predict(x_test)

functions.compute_metrics(y_test, y_pred_rf)

#               precision    recall  f1-score   support

#            0       0.18      0.15      0.16        27
#            1       0.19      0.26      0.22        54
#            2       0.24      0.27      0.25       104
#            3       0.79      0.73      0.76       384

#     accuracy                           0.58       569
#    macro avg       0.35      0.35      0.35       569
# weighted avg       0.60      0.58      0.59       569

# {'precision': 0.6024225425753148, 'recall': 0.5764499121265377, 'f1': 0.5881239122302766}
#               precision    recall  f1-score   support

#            0       1.00      0.00      0.00        27
#            1       0.50      0.02      0.04        54
#            2       0.45      0.18      0.26       104
#            3       0.72      0.98      0.83       384

#     accuracy                           0.70       569
#    macro avg       0.67      0.30      0.28       569
# weighted avg       0.66      0.70      0.61       569

# {'precision': 0.6622077161268726, 'recall': 0.6977152899824253, 'f1': 0.6107531049639207}
