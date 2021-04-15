# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 01:00:17 2021

@author: gauta
"""
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve


def generateConfusionMatrix(test_data, test_prediction):
    cm = confusion_matrix(test_data, test_prediction,labels=[0,1,2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["WithoutMask","WithMask","NotAPerson"])
    disp.plot()
    

def generatePrecisionResult(test_data, test_prediction):
    from sklearn.metrics import precision_score
    precision = precision_score(test_data, test_prediction, average=None,labels=[0,1,2])
    return precision


def generateRecallResult(test_data, test_prediction):
    from sklearn.metrics import recall_score
    recall = recall_score(test_data, test_prediction, average=None,labels=[0,1,2])
    return recall


def generateF1MeasureResult(test_data, test_prediction):
    from sklearn.metrics import f1_score
    f1measure = f1_score(test_data, test_prediction, average=None,labels=[0,1,2])
    return f1measure

def plotRecallPrecisionGraph(test_data,test_prediction):
    classes=[0, 1, 2]    
    plt.figure(figsize=(7, 8))
    
    test_data = label_binarize(test_data, classes=classes)
    test_prediction =label_binarize(test_prediction,classes=classes)
    precision_result = dict()
    recall_result = dict()
    average_precision = dict()
    for i in range(len(classes)):
        precision_result[i], recall_result[i], _ = precision_recall_curve(test_data[:,i], test_prediction[:,i])

    from itertools import cycle
    lines = []
    labels = []
    colors = cycle(['navy', 'darkorange', 'cornflowerblue'])
    
    classDic = {0:'WithoutMask',1:'WithMask',2:'NotAPerson'}

    for i, color in zip(range(len(classes)), colors):
        l, = plt.plot(recall_result[i], precision_result[i])
        lines.append(l)
        labels.append('Precision and recall for class:: {0}'
                      ''.format(classDic[i]))
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision and Recall curve for each classes')
    plt.legend(lines, labels, loc=(0, -.38))
    plt.show()

def evaluateCNNModel(test_data, pred_data):
    precision_result =generatePrecisionResult(test_data, pred_data)
    if len(precision_result) ==3 :
        print("****Precision Metrics****")
        print("Person Without Mask::",round(precision_result[0]*100,2) ,"%")
        print("Person With Mask::",round(precision_result[1] *100,2),"%")
        print("Not a Person::",round(precision_result[2] *100,2),"%")

    print("\n")
    recall_result = generateRecallResult(test_data, pred_data)
    if len(recall_result) ==3 :
        print("****Recall Metrics****")
        print("Person Without Mask::",round(recall_result[0]*100,2),"%")
        print("Person With Mask::",round(recall_result[1]*100,2),"%")
        print("Not a Person::",round(recall_result[2]*100,2),"%")

    print("\n")
    f1_measure_result = generateF1MeasureResult(test_data, pred_data)
    if len(f1_measure_result) ==3 :
        print("****F1 Measure Metrics****")
        print("Person Without Mask::",round(f1_measure_result[0]*100,2),"%")
        print("Person With Mask::",round(f1_measure_result[1]*100,2),"%")
        print("Not a Person::",round(f1_measure_result[2]*100,2),"%")

    print("\n")
    print("****Confusion Metrics****")
    generateConfusionMatrix(test_data, pred_data)
    
    print("****Precision And Recall Graph****")
    plotRecallPrecisionGraph(test_data,pred_data)
        