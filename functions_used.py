import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import os
import acquire

def compute_class_metrics(y_train, y_pred):
    ''' This function will compute ccuracy, true positive rate, false positive rate, true negative rate, 
    false negative rate, precision, recall, f1-score, and support 
    NOTE: needs to input: y_train, y_pred
    '''
    
    counts = pd.crosstab(y_train, y_pred)
    TP = counts.iloc[1,1]
    TN = counts.iloc[0,0]
    FP = counts.iloc[0,1]
    FN = counts.iloc[1,0]
    
    all_ = (TP + TN + FP + FN)

    accuracy = (TP + TN) / all_

    TPR = recall = TP / (TP + FN)
    FPR = FP / (FP + TN)

    TNR = TN / (FP + TN)
    FNR = FN / (FN + TP)

    precision =  TP / (TP + FP)
    f1 =  2 * ((precision * recall) / ( precision + recall))

    support_pos = TP + FN
    support_neg = FP + TN
    
    # print(f"Accuracy: {accuracy}\n")
    print(f"True Positive Rate/Sensitivity/Recall/Power: {TPR}")
    print(f"False Positive Rate/False Alarm Ratio/Fall-out: {FPR}")
    print(f"True Negative Rate/Specificity/Selectivity: {TNR}")
    print(f"False Negative Rate/Miss Rate: {FNR}\n")
    # print(f"Precision/PPV: {precision}")
    # print(f"F1 Score: {f1}\n")
    # print(f"Support (0): {support_pos}")
    # print(f"Support (1): {support_neg}")
    
    # printing the classification report
    print(classification_report(y_train, y_pred))
    
def decision_tree(X_train, y_train, X_validate, y_validate):
    '''
    This will use Decision Tree modeling 
    This function will calculate teh accuracy score and validation score given with max depth up to 10
    send in X_train, y_train, X_validate, y_validate
    '''
    for x in range(1,11):
        #create the object
        tree = DecisionTreeClassifier(max_depth=x)

        #fit the object
        # only using TRAIN DATA in this!
        tree.fit(X_train, y_train) 

        # calculate the accuracy for train
        accuracy = tree.score(X_train, y_train)

        # calculate the accuracy for validate
        accuracy_validate = tree.score(X_validate, y_validate)

        # printing result for x, accuracy, validation from validate_data_set
        print(f'max depth of {x}, the accuracy score = {round(accuracy,2)}, validate score = {round(accuracy_validate,2)}')
        
def random_forest(X_train, y_train, X_validate, y_validate):
    '''
    This function will use Random Forest modeling 
    The ouput of of this function is min leaf samples, max depth, accuracy score, and validate score
    send in X_train, y_train, X_validate, y_validate
    '''
    for x in range(1,11):
        rf = RandomForestClassifier(min_samples_leaf=x, max_depth=11-x, random_state=123)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_train)
        accuracy = rf.score(X_train, y_train)
        accuracy_validation = rf.score(X_validate, y_validate)
           # stats.append([ x, 11-x, round(accuracy,2), round(accuracy_validation,2) ]) 
        print(f'min_leaf_samples = {x}, max_depth = {11-x}, accuracy score = {round(accuracy, 2)}, validate score {round(accuracy_validation,2)}')
        
def lr_score(X_train, y_train, X_validate, y_validate):
    '''
    This function provide accuracy scores and validate score for a given features
    The function will have 4 features given
    Output is accuracy score and validation score of each features given
    '''
    lr = LogisticRegression()
    
    features1 = ['monthly_charges', 'contract_type_One year', 'contract_type_Two year']
    features2 = ['monthly_charges', 'contract_type_One year', 'contract_type_Two year', 'paperless_billing_Yes']
    features3 = ['monthly_charges', 'contract_type_One year', 'contract_type_Two year', 'paperless_billing_Yes', 
             'internet_service_type_Fiber optic']
    features4 = ['monthly_charges', 'contract_type_One year', 'contract_type_Two year', 'paperless_billing_Yes', 
             'internet_service_type_Fiber optic', 'phone_service_Yes']
    lr.fit(X_train[features1], y_train)
    lr1_score = lr.score(X_train[features1], y_train)
    lr1_validate = lr.score(X_validate[features1], y_validate)
    
    lr.fit(X_train[features2], y_train)
    lr2_score = lr.score(X_train[features2], y_train)
    lr2_validate = lr.score(X_validate[features2], y_validate)
    
    lr.fit(X_train[features3], y_train)
    lr3_score = lr.score(X_train[features3], y_train)
    lr3_validate = lr.score(X_validate[features3], y_validate)
    
    lr.fit(X_train[features4], y_train)
    lr4_score = lr.score(X_train[features4], y_train)
    lr4_validate = lr.score(X_validate[features4], y_validate)
    
    print(f'accuracy_score with features 1: {round(lr1_score,3)}, validation_score: {round(lr1_validate,3)}')
    print(f'accuracy_score with features 2: {round(lr2_score,3)}, validation_score: {round(lr2_validate,3)}')
    print(f'accuracy_score with features 3: {round(lr3_score,3)}, validation_score: {round(lr3_validate,3)}')
    print(f'accuracy_score with features 4: {round(lr4_score,3)}, validation_score: {round(lr4_validate,3)}')
    
def check_file_churn(filename, y_pred, y_pred_proba):
    ''' 
    This checks if a file does exist or not from directory, if not it read from mySQL
    Input: filename: 'churn_csv', df_y, y_pred_proba, df_customer_id
    '''
    df_y_pred = pd.DataFrame(y_pred)
    df_y_pred_proba = pd.DataFrame(y_pred_proba)  
    df_customer_id = acquire.get_telco_db()['customer_id']
    
    df_y = pd.merge(df_y_pred_proba, df_y_pred, left_index=True, right_index=True)
    df_y = pd.merge(df_customer_id, df_y, left_index=True, right_index=True)
    
    df_y = df_y.rename(columns={1:'Probability of Churn', '0_y':'Prediction of Churn'})
    df_y = df_y.drop(columns={'0_x'})
    
    filename = 'churn_csv'
    
    if os.path.exists(filename):
        print('this file exists, reading from csv file')
        churn_csv = pd.read_csv('churn_csv')
    else:
        print('this file does not exist, export file into csv')
    churn_csv = df_y.to_csv('churn_csv', index=False)
    print(pd.read_csv(filename))
    
    return churn_csv