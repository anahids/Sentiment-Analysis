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
    REPLACE_NO_SPACE = re.compile("[*-.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews

# Converts each review to a numeric representation
def vectorization(reviews_train_clean, reviews_test_clean):
    stop_words = ['in', 'of', 'at', 'a', 'the', 'an']
    cv = CountVectorizer(binary=True, stop_words=stop_words)
    cv.fit(reviews_train_clean)
    X = cv.transform(reviews_train_clean)
    X_test = cv.transform(reviews_test_clean)
    return cv, X, X_test

def classifierRegularization(X, target):
    X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75) 
    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        lr = LogisticRegression(C=c, solver='liblinear')
        lr.fit(X_train, y_train)
        print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))
    
def trainingModel(X, X_Test, target):
    final_model = LogisticRegression(C=0.05, solver='liblinear')
    final_model.fit(X, target)
    predictions = final_model.predict(X_Test)
    return final_model, list(predictions)

# The classifier object contains the most informative words that it obtained during analysis. These words basically have a strong say in what’s classified as a positive or a negative review
def predictors(cv, final_model):
    feature_to_coef = { word: coef for word, coef in zip(cv.get_feature_names(), final_model.coef_[0]) }

    for best_positive in sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=True)[:5]:
        print (best_positive)
    
    for best_negative in sorted(feature_to_coef.items(), key=lambda x: x[1])[:5]:
        print (best_negative)

def main():
    print("Choose an option:")
    print("1 Use the data set test of reviews")
    print("2 Use a new review ")
    option = input()

    reviews_train = openTraining()
    reviews_train_clean = clean(reviews_train)
    target = [1 if i < 12500 else 0 for i in range(25000)] # Positive and Negative

    if option == "1":
        reviews_test = openFullTest()
        reviews_test_clean = clean(reviews_test)
        cv, X, X_Test = vectorization(reviews_train_clean,reviews_test_clean)
        classifierRegularization(X, target)
        final_model, predictions = trainingModel(X, X_Test, target)
        print ("\nFinal Accuracy: %s" % accuracy_score(target, predictions))
        #print("\nThe predictions are:")
        #print(predictions)
        predictors(cv, final_model)

    elif option == "2" :
        review = input('\nInsert review: ')
        reviewL = [review]
        review_clean = clean(reviewL)
        cv, X, X_Test = vectorization(reviews_train_clean,review_clean)
        final_model, predictions = trainingModel(X, X_Test, target)
        print("\nThe prediction for %s is: %s" % (review, predictions))

if __name__ == "__main__":
    main()