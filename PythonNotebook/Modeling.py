from sklearn.model_selection import train_test_split
x=data.drop(["Class"],axis=1)
y=data["Class"]
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
model_log = LogisticRegression()
model_log.fit(x_train ,y_train)
y_pred_log = model_log.predict(x_test)
print(f"The accuracy of this model is {round(model_log.score(x_test,y_test)*100,2)}%")

#F1 AND CONFUSION MATRIX
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score,precision_score,recall_score,classification_report
print(f"F1-Score of this model is {round(f1_score(y_test , y_pred_log),2)}")
with plt.style.context(('ggplot')):
    sns.heatmap(confusion_matrix(y_test,y_pred_log),annot = True)
    plt.show()

#RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train,y_train)
y_pred_random = model.predict(x_test)
print(f"the accuracy of this model is {round(model.score(x_test,y_test)*100,2)}%")

#GENETIC SELECTION
! pip install sklearn-genetic
from genetic_selection import GeneticSelectionCV
estimator_log = LogisticRegression()
model_ga_log = GeneticSelectionCV(estimator = estimator_log,cv=5,scoring='accuracy',max_features=6,n_population=60,n_gen_no_change=5)
model_ga_log.fit(x_train , y_train)
y_log_ga = model_ga_log.predict(x_test)

#XGBOOST
! pip install xgboost
from xgboost import XGBClassifier
model_xgb = XGBClassifier(max_depth=2)
model_xgb.fit(x_train,y_train)
y_pred_xgb = model_xgb.predict(x_test)

#PREDICTION
from sklearn import metrics
from sklearn.metrics import roc_curve,roc_auc_score

#XGBoost
fpr_xgb , tpr_xgb ,_ = metrics.roc_curve(y_test,y_pred_xgb)
auc_xgb = metrics.roc_auc_score(y_test , y_pred_xgb)
#Logistic regression
fpr_log , tpr_log ,_ = metrics.roc_curve(y_test,y_pred_log)
auc_log = metrics.roc_auc_score(y_test , y_pred_log)
#Random Forest
fpr_ran , tpr_ran ,_ = metrics.roc_curve(y_test,y_pred_random)
auc_ran = metrics.roc_auc_score(y_test , y_pred_random)
#Genetic + Logistic
fpr_log_ga , tpr_log_ga ,_ = metrics.roc_curve(y_test,y_log_ga)
auc_log_ga = metrics.roc_auc_score(y_test , y_log_ga)

with plt.style.context(('ggplot')):
    plt.figure(figsize=(10,7))
    plt.title("ROC Curve")
    plt.plot(fpr_xgb,tpr_xgb,label="AUC_XGB="+str(auc_xgb))
    plt.plot(fpr_log,tpr_log,label="AUC_logistic_regression="+str(auc_log))
    plt.plot(fpr_ran,tpr_ran,label="AUC_random_forest="+str(auc_ran))
    plt.plot(fpr_log_ga,tpr_log_ga,label="AUC_genetic_logistic="+str(auc_log_ga))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()
    
