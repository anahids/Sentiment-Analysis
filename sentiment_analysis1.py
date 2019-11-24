import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Open training movie data
def openTraining():
    reviews_train = []
    for line in open('movie_data/full_train.txt', 'r'):
        reviews_train.append(line.strip())
    return reviews_train

# Open full test movie data
def openFullTest():
    reviews_test = []
    for line in open('movie_data/full_test.txt', 'r'):
        reviews_test.append(line.strip())
    return reviews_test

# Clean data
def clean(reviews):
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]

    return reviews

def classifier():
    target = [1 if i < 12500 else 0 for i in range(25000)]

    X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        lr = LogisticRegression(C=c)
        lr.fit(X_train, y_train)
        print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))

    final_model = LogisticRegression(C=0.05)
    final_model.fit(X, target)
    print ("Final Accuracy: %s" % accuracy_score(target, final_model.predict(X_test)))
    return final_model

def predictors():
    feature_to_coef = { word: coef for word, coef in zip(cv.get_feature_names(), final_model.coef_[0]) }

    for best_positive in sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=True)[:5]:
        print (best_positive)
    
    for best_negative in sorted(feature_to_coef.items(), key=lambda x: x[1])[:5]:
        print (best_negative)

if __name__ == "__main__":
    reviews_train = openTraining()
    reviews_test = openFullTest()
    reviews_train_clean = clean(reviews_train)
    reviews_test_clean = clean(reviews_test)

    cv = CountVectorizer(binary=True)
    cv.fit(reviews_train_clean)
    X = cv.transform(reviews_train_clean)
    X_test = cv.transform(reviews_test_clean)

    final_model = classifier()
    predictors()
