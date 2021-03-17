from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


def containWord(sentence, word):
    if isinstance(sentence, str):
        if word in sentence.split():
            return 1
    return 0


def get_words_length(sentence):
    if isinstance(sentence, str):
        return len(sentence.split())
    return 0


def get_word_counts(df):
    counts = []
    for sentence in df['Text'].values:
        if isinstance(sentence, str):
            counts.append(len(sentence.split()))
        else:
            counts.append(0)
    return counts


def get_top_words(df):
    words_map = {}
    stop = set(stopwords.words('english'))
    for sentence in df['Text'].values:
        if isinstance(sentence, str):
            fredist = nltk.FreqDist(sentence.split())
            for localkey in fredist.keys():
                if localkey in stop:
                    continue
                if localkey in words_map.keys():
                    words_map[localkey] = words_map[localkey] + fredist[localkey]
                else:
                    words_map[localkey] = fredist[localkey]
    sub_count_list = list(words_map.items())
    sub_count_list.sort(key=lambda x: x[1], reverse=True)
    return sub_count_list[0:50]


def get_words(df):
    words_2 = get_top_words(df[df["Score"] == 2])
    words_1 = get_top_words(df[df["Score"] == 1])
    words_3 = get_top_words(df[df["Score"] == 3])
    words_4 = get_top_words(df[df["Score"] == 4])
    words_5 = get_top_words(df[df["Score"] == 5])
    words = [item for item, value in words_1]
    words = set(words)
    for item, value in words_2:
        words.add(item)
    for item, value in words_3:
        words.add(item)
    for item, value in words_4:
        words.add(item)
    for item, value in words_5:
        words.add(item)
    return words


# Load files into DataFrames
X_train = pd.read_csv("./data/X_train.csv")
X_submission = pd.read_csv("./data/X_submission.csv")

print("Load File")
# Process the DataFrames

X_train['Count'] = X_train.apply(lambda x: get_words_length(x.Text), axis=1)
X_submission['Count'] = X_submission.apply(lambda x: get_words_length(x.Text), axis=1)

for word in get_words(X_train):
    key = "contain_" + word
    X_train[key] = X_train.apply(lambda x: containWord(x.Text, word), axis=1)
    X_submission[key] = X_submission.apply(lambda x: containWord(x.Text, word), axis=1)

print("Finish Features")

# Split training set into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(
    X_train.drop(['Score'], axis=1),
    X_train['Score'],
    test_size=1 / 4.0,
    random_state=0
)

X_train_processed = X_train.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary'])
X_test_processed = X_test.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary'])
X_submission_processed = X_submission.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'Score'])

print("Start Modeling")

# Learn the model
#   model = KNeighborsClassifier(n_neighbors=3).fit(X_train_processed, Y_train)
# model = SVC(gamma='auto', C=5, kernel="linear").fit(X_train_processed, Y_train)
model = RandomForestRegressor().fit(X_train_processed, Y_train)

print("Start Predict")

# Predict the score using the model
Y_test_predictions = model.predict(X_test_processed)
X_submission['Score'] = model.predict(X_submission_processed)

# Evaluate your model on the testing set
print("RMSE on testing set = ", mean_squared_error(Y_test, Y_test_predictions))

# Create the submission file
submission = X_submission[['Id', 'Score']]
submission.to_csv("./data/submission.csv", index=False)
