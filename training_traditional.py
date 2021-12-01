# -------------------------------------------------------------------------
# Implements training for traditional models functionality from eRisk model
#
# email ecampillo@lsi.uned.es
# -------------------------------------------------------------------------

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import naive_bayes, ensemble
import sklearn.metrics as metrics    
import xgboost
from utils import *


def train_and_evaluate(X_train, y_train, X_test, y_test, train_weights, classifier_name="svm", strategy="weights"):
    
    logger("Starting training traditional")
    classifier = train(X_train, y_train, train_weights, classifier_name, strategy)
    y_pred = evaluate(classifier, X_test, y_test)
    
    return y_pred, classifier
    

def train(x_train, y_train, train_weights, classifier_name="svm", strategy="weights"):

    train_feats = x_train
    train_labels = y_train

    if strategy == 'weights':
        strategy=None
    elif strategy == 'normal':
        strategy=None
        train_weights = None
    else:
        train_weights=None

    if classifier_name == "svm":
        classifier = SVC(class_weight=strategy)
    elif classifier_name == "linear_svm":
        classifier = LinearSVC(class_weight=strategy)
    elif classifier_name == "forest":
        classifier = ensemble.RandomForestClassifier(class_weight=strategy)
    elif classifier_name == "xgboost":
        classifier = xgboost.XGBClassifier(class_weight=strategy)
    else:
        classifier = naive_bayes.MultinomialNB()

    classifier.fit(train_feats, train_labels, sample_weight=train_weights)
    
    return classifier

def evaluate(classifier, x_test, y_test):

    test_feats = x_test
    y_pred = classifier.predict(test_feats)
    
    classification_report = metrics.classification_report(y_test, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    
    logger(classification_report)
    logger(confusion_matrix)
    
    return y_pred