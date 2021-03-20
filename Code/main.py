
import torch
import os
import cv2
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
import constants
from sklearn.metrics import confusion_matrix

classes=[constants.Masked,constants.UnMasked,constants.NonPerson]

DATA_REBUILD=False


def generateConfusionMatrix(test_data,test_prediction):
    print("Confusion Matrix:\n", confusion_matrix(test_data, test_prediction))

def generateAccuracyResult(test_data,cnn):
    correct = 0
    total = 0
    test_data_size =len(test_data)
    with torch.no_grad():
        for images, labels in test_data:
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * (correct / total)
    return accuracy
    
def generatePrecisionResult(test_data,test_prediction):
    from sklearn.metrics import precision_score
    precision = precision_score(test_data, test_prediction, average=None)
    return precision
    
def generateRecallResult(test_data,test_prediction):
    from sklearn.metrics import recall_score
    recall = recall_score(test_data, test_prediction, average=None)
    return recall

def generateF1MeasureResult(precision,recall):
    f1measure = 2 * (precision * recall) / (precision + recall)
    return f1measure

    
class ImageClassifier():
    ImageDimensions=constants.imageSize
    trained_Data=[]
    Labels={constants.NonPerson:2,constants.Masked:1,constants.UnMasked:0}
    
    def ImageDataSetPrep(self):
        base_Path=Path(constants.Base_Path)
        print('path',constants.Base_Path)
        masked_Path=base_Path/constants.Masked_Folder
        unmasked_Path=base_Path/constants.UnMasked_Folder
        nonPerson_Path=base_Path/constants.NonPerson_Folder
        path_dirs = [ [nonPerson_Path,2],[masked_Path,1],[unmasked_Path,0] ] 
        if not os.path.exists(base_Path):
            raise Exception("The data path doesn't exist")
        for Dir_path,label in path_dirs:
            print("Current Label is ",label)
            for image in os.listdir(Dir_path):
                imagePath=os.path.join(Dir_path,image)
                try:
                    img = cv2.imread(imagePath)
                    img = cv2.resize(img, (self.ImageDimensions,self.ImageDimensions))
                    self.trained_Data.append([np.array(img, dtype="object"), label])
                except:
                    print('Error Occured while processing images')
                    pass
        np.random.shuffle(self.trained_Data)
        now=datetime.now()
        dt_string = now.strftime("%d%m%Y%H%M%S")
        path=constants.training_dataSavedPath+dt_string+constants.trained_datasetName
        np.save(path, self.trained_Data)
            
    
if DATA_REBUILD:
    batchData=ImageClassifier()
    batchData.ImageDataSetPrep()
else:
    val = os.chdir(constants.training_dataSavedPath)
    files = glob.glob(os.getcwd()+'/*')
    #files = glob.glob(constants.training_dataSavedPath+'*')
    latest_file = max(files, key=os.path.getctime)
    print(latest_file)
        
    
    
    