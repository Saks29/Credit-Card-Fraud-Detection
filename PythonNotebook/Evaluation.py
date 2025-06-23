y_pred = model_xgb.predict(x_test)

# Display the evaluation results
print(f"Accuracy:- {round(accuracy_score(y_test, y_pred)*100,2)}%")
print(f"Precision:- {round(precision_score(y_test, y_pred, average='macro', zero_division=0)*100,2)}%")
print(f"Recall:- {round(recall_score(y_test, y_pred, average='macro', zero_division=0)*100,2)}%")
print(f"Classification report:- {classification_report(y_test, y_pred)}")

""" Accuracy:- 99.91%
Precision:- 97.28%
Recall:- 90.73%
Classification report:-               precision    recall  f1-score   support

           0       1.00      1.00      1.00     27809
           1       0.95      0.81      0.88       108

    accuracy                           1.00     27917
   macro avg       0.97      0.91      0.94     27917
weighted avg       1.00      1.00      1.00     27917
