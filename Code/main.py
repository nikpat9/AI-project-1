import torch
import os
import cv2
from path import Path
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from Code import constants
import gc


def ImageDataSetPrep(data_type):
    if data_type == "test":
        path_test_train = constants.TEST
        base_Path = Path(constants.Base_Path)
        masked_Path = base_Path / constants.Masked_Folder_test
        unmasked_Path = base_Path / constants.UnMasked_Folder_test
        nonPerson_Path = base_Path / constants.NonPerson_Folder_test
    else:
        path_test_train = constants.Train
        base_Path = Path(constants.Base_Path)
        masked_Path = base_Path / constants.Masked_Folder_train
        unmasked_Path = base_Path / constants.UnMasked_Folder_train
        nonPerson_Path = base_Path / constants.NonPerson_Folder_train

    if len(os.listdir(masked_Path)) != 0 or len(os.listdir(unmasked_Path)) != 0 or len(os.listdir(nonPerson_Path)) != 0:
        path_dirs = [[nonPerson_Path, 2], [masked_Path, 1], [unmasked_Path, 0]]
        if not os.path.exists(base_Path):
            raise Exception("The data path doesn't exist")
        for Dir_path, label in path_dirs:
            for image in os.listdir(Dir_path):
                imagePath = os.path.join(Dir_path, image)
                try:
                    img = cv2.imread(imagePath)
                    img = cv2.resize(img, (constants.imageSize, constants.imageSize))
                    if label == 2:
                        cv2.imwrite(path_test_train + "\class2\\" + image + ".jpg", img)
                    if label == 1:
                        cv2.imwrite(path_test_train + "\class1\\" + image + ".jpg", img)
                    if label == 0:
                        cv2.imwrite(path_test_train + "\class0\\" + image + ".jpg", img)
                except:
                    print('Error Occured while processing images')
                    pass


def generateConfusionMatrix(test_data, test_prediction):
    print("Confusion Matrix:\n", confusion_matrix(test_data, test_prediction,labels=[0,1,2]))


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


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(100, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4

            nn.Flatten(),
            nn.Linear(160000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3))

    def forward(self, x):
        return self.cnn_layers(x)


if __name__ == '__main__':
    ImageDataSetPrep("test")
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    testset = torchvision.datasets.ImageFolder(root=constants.TEST, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=8)
    PATH = "state_dict_model.pt"
    if not os.path.exists("state_dict_model.pt"):
        ImageDataSetPrep("train")
        trainset = torchvision.datasets.ImageFolder(root=constants.Train, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
        for epoch in range(constants.EPOCH):
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = net.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                gc.collect()

        # Save
        torch.save(net, PATH)
    else:
        net = torch.load(PATH)
    pred_data = []
    test_data = []
    correct = 0
    total = 0
    
    for i, data in enumerate(testloader, 0):
        images, labels = data
        outputs = net(images)
        test_data.extend(labels.numpy().tolist())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        pred_data.extend(predicted.numpy().tolist())
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on ',len(test_data), ' test images: %d %%' % (
            100 * correct / total))
    
    print("Confusion Matrix :: [PersonWithoutMask  PersonWithMask  NotAPerson]")
    generateConfusionMatrix(test_data, pred_data)
    
    precision_result =generatePrecisionResult(test_data, pred_data)
    if len(precision_result) ==3 :
        print("****Precision Metrics****")
        print("Person Without Mask::",precision_result[0])
        print("Person With Mask::",precision_result[1])
        print("Not a Person::",precision_result[2])

    recall_result = generateRecallResult(test_data, pred_data)
    if len(recall_result) ==3 :
        print("****Recall Metrics****")
        print("Person Without Mask::",recall_result[0])
        print("Person With Mask::",recall_result[1])
        print("Not a Person::",recall_result[2])
    
    f1_measure_result = generateF1MeasureResult(test_data, pred_data)
    if len(f1_measure_result) ==3 :
        print("****F1 Measure Metrics****")
        print("Person Without Mask::",f1_measure_result[0])
        print("Person With Mask::",f1_measure_result[1])
        print("Not a Person::",f1_measure_result[2])
