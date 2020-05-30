
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC, SVC
from sklearn import svm
# import pipeline class
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.metrics import roc_curve, auc,precision_recall_curve,average_precision_score,roc_auc_score
import numpy as np
# use MultinomialNB algorithm
from sklearn.naive_bayes import MultinomialNB
# import method for split train/test data set
from sklearn.model_selection import train_test_split
# import method to calculate metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
# import GridSearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import numpy as np
from matplotlib import pyplot as plt
from statistics import mean
from sklearn.metrics import confusion_matrix 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 


# Q1

def classify(train_file, test_file):
    
    data=pd.read_csv(train_file,header=0)
    test=pd.read_csv(test_file,header=0)

    text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                         ('clf', svm.LinearSVC())])

    parameters = {'tfidf__min_df':[1, 2,3],
                  'tfidf__stop_words':[None,"english"],
                  'clf__C': [0.5,1.0,5.0]}

    metric =  "f1_macro"
    
    gs_clf = GridSearchCV(text_clf, param_grid=parameters, scoring=metric, cv=6)
    
    gs_clf = gs_clf.fit(data["text"],data["label"])
    
    for param_name in gs_clf.best_params_:
        print(param_name,": ",gs_clf.best_params_[param_name])
    print("best f1 score:", gs_clf.best_score_)
    
    
    #
    
    tfidf_vect = TfidfVectorizer(stop_words=None, min_df=1) 
    
    dtm=tfidf_vect.fit_transform(data["text"])
    dtm2 = tfidf_vect.transform(test["text"])
    
    x_train=dtm
    y_train=data["label"]
    x_test=dtm2
    y_test=test["label"]
    clf=svm.LinearSVC(C = 0.5).fit(x_train, y_train)
    predicted=clf.predict(x_test)
    labels=sorted(data["label"].unique())
    
    precision, recall, fscore, support=         precision_recall_fscore_support(         y_test, predicted, labels=labels)
    print("\n")
    print("labels: ", labels)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f-score: ", fscore)
    print("support: ", support)
    
    predict_p=clf.decision_function(dtm2)
    
    print("\n")
    print("Area under curve:", roc_auc_score(test["label"],predict_p))
    fpr,tpr,th=roc_curve(test["label"],predict_p, pos_label=1)
    plt.figure();
    plt.plot(fpr,tpr,color='darkorange',lw=2);
    plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--');
    plt.xlim([0.0,1.0])
    plt.ylim([0.0, 1.05]);
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate');
    plt.title('AUC of Naive Bayes Model');
    plt.show();


    print("average precision : ",average_precision_score(test["label"],predict_p))
    precision, recall, thresholds = precision_recall_curve(test["label"],                                     predict_p, pos_label=1)
    plt.figure();
    plt.plot(recall, precision, color='darkorange', lw=2);
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05]);
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    plt.title('Precision_Recall_Curve of Naive Bayes Model');
    plt.show();

    return None


# Q2

def K_fold_CV(train_file):
    
    data = pd.read_csv(train_file,header=0)
    i=2
    tfidf_vect = TfidfVectorizer() 
    dtm= tfidf_vect.fit_transform(data["text"])
    metrics = ["roc_auc"]
    clf = MultinomialNB()
    clf2 = svm.LinearSVC()
    MNB=list()
    SVM=list()

    while i <= 20:
        cv = cross_validate(clf, dtm, data["label"],                         scoring=metrics, cv=i,                         return_train_score=True)
        MNB.append((mean(cv["test_roc_auc"])))

        cv2 = cross_validate(clf2, dtm, data["label"],                         scoring=metrics, cv=i)

        SVM.append((mean(cv2["test_roc_auc"])))
        i+=1
        
    
    plt.figure();
    plt.plot(list(range(2,21)), MNB, color='darkorange', lw=2);
    plt.plot(list(range(2,21)), SVM, color='navy', lw=2);
    plt.xlim([2.0, 20.0]);
    #plt.ylim([0.0, 1.05]);
    plt.xlabel('k-fold');
    plt.ylabel('AUC');
    plt.title('MultinomialNB vs LinearSVC ');
    plt.show();
    return None



if __name__ == "__main__":  
    # Question 1
    classify("assign4_train.csv",               r"assign4_test.csv")
    #classify(r"assign4_train.csv","assign4_test.csv")
    
    # Test Q2
    print("\nQ2")
    K_fold_CV(r"assign4_train.csv")
   