import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Open training movie data
def openTraining():
    return [line.strip() for line in open('movie_data/full_train.txt', 'r')]

# Open full test movie data
def openFullTest():
    return [line.strip() for line in open('movie_data/full_test.txt', 'r')]

# Clean data
def clean(reviews):
    replaceNS = re.compile("[*-.;:!\'?,\"()\[\]]")
    replaceWS = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    reviews = [replaceNS.sub("", review.lower()) for review in reviews]
    reviews = [replaceWS.sub(" ", review) for review in reviews]
    return reviews

# Converts each review to a numeric representation
def vectorization(reviewsTrainClean,reviewsTestClean):
    stopWords = ['in', 'of', 'at', 'a', 'the', 'an']
    classVect = CountVectorizer(binary=True, stop_words=stopWords)
    classVect.fit(reviewsTrainClean)
    X = classVect.transform(reviewsTrainClean)
    X_test = classVect.transform(reviewsTestClean)
    return classVect, X, X_test

def classifierRegularization(X, target):
    X_train, X_fVal, y_train, y_fVal = train_test_split(X, target, train_size = 0.75) 
    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        lr = LogisticRegression(C=c, solver='liblinear')
        lr.fit(X_train, y_train)
        print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_fVal, lr.predict(X_fVal))))
    
def trainingModel(X, X_Test, target):
    fModel = LogisticRegression(C=0.05, solver='liblinear')
    fModel.fit(X, target)
    predictions = fModel.predict(X_Test)
    return fModel, list(predictions)

# The classifier object contains the most informative words that it obtained during analysis. These words basically have a strong say in what’s classified as a positive or a negative review
def predictors(classVect, fModel):
    featureCoef = { word: coef for word, coef in zip(classVect.get_feature_names(), fModel.coef_[0]) }

    for bPositive in sorted(featureCoef.items(), key=lambda x: x[1], reverse=True)[:5]:
        print (bPositive)
    
    for bNegative in sorted(featureCoef.items(), key=lambda x: x[1])[:5]:
        print (bNegative)

def checkPolarity(predictions):
    for prediction in predictions:
        sentiment = "Positive" if prediction == 1 else "Negative" 
    return sentiment

def main():
    print("Choose an option:")
    print("1 Use the data set test of reviews")
    print("2 Use a new review ")
    option = input()

    reviewsTrain = openTraining()
    reviewsTrainClean = clean(reviewsTrain)
    target = [1 if i < 12500 else 0 for i in range(25000)] # Positive and Negative

    if option == "1":
        reviewsTest = openFullTest()
        reviewsTestClean = clean(reviewsTest)
        classVect, X, X_Test = vectorization(reviewsTrainClean,reviewsTestClean)
        classifierRegularization(X, target)
        fModel, predictions = trainingModel(X, X_Test, target)
        #classifierRegularization(X, target)
        print ("\nFinal Accuracy: %s\n" % accuracy_score(target, predictions))
        print("\nThe predictions are:")
        print(predictions)
        #predictors(classVect, fModel)

    elif option == "2" :
        review = input('\nInsert review: ')
        reviewL = [review]
        reviewClean = clean(reviewL)
        classVect, X, X_Test = vectorization(reviewsTrainClean,reviewClean)
        fModel, predictions = trainingModel(X, X_Test, target)
        polarity = checkPolarity(predictions)
        print("\nThe polarity for %s is: %s" % (review, polarity))

if __name__ == "__main__":
    main()