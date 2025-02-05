
from sklearn.metrics import confusion_matrix,\
mean_squared_error,f1_score
from model import build_model
import numpy as np
import matplotlib.pyplot as plt

def calculate_matrics(conf_matrix):
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1_score = 2 * (precision*recall)/ (precision + recall)
    
    return {
        'Accuracy': accuracy, 'Precision': precision,'Recall': recall,
        'F1 Score': f1_score, 'Specificity': specificity, 'True Positives': TP,
        'True Negatives': TN, 'False Positives': FP, 'False Negatives': FN
            }

def AUC_ROC(Recall, Specificity):
    plt.figure(figsize=(8, 5))
    plt.plot(Recall, Specificity, 'o', color='red', markersize= 8)
    plt.title("Recall'TPR' against Specificity'FPR'")
    plt.xlabel('Recall')
    plt.ylabel('Specificity')
    plt.show()
