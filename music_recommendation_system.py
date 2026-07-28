#Music Recommendation System
# Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Loading
data = pd.read_csv("spotify_history_multiuser.csv")
print(data.head())

# data exploring
data.info()
print("\n")
print(data.describe())

# data cleaning
print("Missing values:\n", data.isnull().sum())
data = data.dropna()
data = data.drop_duplicates()

# data preprocessing
data['ts'] = pd.to_datetime(data['ts']) 
data = data.sort_values(['user_id', 'track_name', 'ts']).reset_index(drop=True)

# extract useful features from the timestamp
data['hour'] = data['ts'].dt.hour
data['day_of_week'] = data['ts'].dt.dayofweek

# how many times this user has played this track before (so far)
data['play_count_so_far'] = data.groupby(['user_id', 'track_name']).cumcount()

# convert into 0-1
data['shuffle'] = data['shuffle'].astype(int)
data['skipped'] = data['skipped'].astype(int)

# Encode text categories into numbers
encoder = LabelEncoder()
data['reason_start'] = encoder.fit_transform(data['reason_start'].astype(str))
data['reason_end'] = encoder.fit_transform(data['reason_end'].astype(str))
data['user_id_enc'] = encoder.fit_transform(data['user_id'])

# Features x and y
X = data[['ms_play', 'shuffle', 'skipped', 'hour', 'day_of_week',
          'play_count_so_far', 'reason_start', 'reason_end', 'user_id_enc']]
y = data['replayed_within_30_days']

# Normalize features to a 0-1 range
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# model train and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Logistic Regression
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

print("Logistic Regression:")
print("Accuracy:", round(accuracy_score(y_test, y_pred_lr),4))
print("Precision: ", round(precision_score(y_test, y_pred_lr),4))
print("Recall:", round(recall_score(y_test, y_pred_lr),4))
print("F1-Score:", round(f1_score(y_test, y_pred_lr),4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# Model 2: Decision Tree
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

print("\nDecision Tree:")
print("Accuracy:", round(accuracy_score(y_test, y_pred_dt),4))
print("Precision: ", round(precision_score(y_test, y_pred_dt),4))
print("Recall:", round(recall_score(y_test, y_pred_dt),4))
print("F1-Score:", round(f1_score(y_test, y_pred_dt),4))

# Model 3: Random Forest
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

print("\nRandom Forest")
print("Accuracy:", round(accuracy_score(y_test, y_pred_rf),4))
print("Precision: ", round(precision_score(y_test, y_pred_rf),4))
print("Recall:", round(recall_score(y_test, y_pred_rf),4))
print("F1-Score:", round(f1_score(y_test, y_pred_rf),4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Model 4: KNN
model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)

print("\nKNN")
print("Accuracy:", round(accuracy_score(y_test, y_pred_knn),4))
print("Precision: ", round(precision_score(y_test, y_pred_knn),4))
print("Recall:", round(recall_score(y_test, y_pred_knn),4))
print("F1-Score:", round(f1_score(y_test, y_pred_knn),4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))

# Predict probability for every track using the best model (Random Forest)
data['predicted_replay_prob'] = model_rf.predict_proba(X)[:, 1]

# Showing top 5 recommended tracks for each user
for user in data['user_id'].unique():
    top_tracks = data[data['user_id'] == user].sort_values(
        'predicted_replay_prob', ascending=False
    )[['track_name', 'artist_name', 'predicted_replay_prob']].head(5)

    print(f"\nTop 5 recommended tracks for {user}:")
    print(top_tracks)