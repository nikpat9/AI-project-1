import torch
import os
import cv2
from path import Path
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision
from torch import nn, optim
from Code import constants
from Code import metricsEvaluation
import gc
import pandas as pd
from sklearn.model_selection import StratifiedKFold

TESTMALE = "../Images/testMale"
TESTFEMALE = "../Images/testFemale"

def ImageDataSetPrep(data_type):
    if data_type == "test":
        path_test_train = constants.TEST
        base_Path = Path(constants.Base_Path)
        masked_Path = base_Path + "/" + constants.Masked_Folder_test
        unmasked_Path = base_Path + "/" + constants.UnMasked_Folder_test
        nonPerson_Path = base_Path + "/" + constants.NonPerson_Folder_test
    else:
        path_test_train = constants.Train
        base_Path = Path(constants.Base_Path)
        masked_Path = base_Path + "/" + constants.Masked_Folder_train
        unmasked_Path = base_Path + "/" + constants.UnMasked_Folder_train
        nonPerson_Path = base_Path + "/" + constants.NonPerson_Folder_train

    if len(os.listdir(masked_Path)) != 0 or len(os.listdir(unmasked_Path)) != 0 or len(os.listdir(nonPerson_Path)) != 0:
        path_dirs = [[nonPerson_Path, 2], [masked_Path, 1], [unmasked_Path, 0]]
        if not os.path.exists(base_Path):
            raise Exception("The data path doesn't exist")
        for Dir_path, label in path_dirs:
            for image in os.listdir(Dir_path):
                imagePath = os.path.normpath(Dir_path)
                imagePath = os.path.join(imagePath, image)
                try:
                    img = cv2.imread(imagePath)
                    img = cv2.resize(img, (constants.imageSize, constants.imageSize))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    tempImg = img
                    tempImage = os.path.splitext(image)[0]
                    image = tempImage
                    if label == 2:
                        cv2.imwrite(path_test_train + "\class2\\" + image + ".png", tempImg)
                    if label == 1:
                        cv2.imwrite(path_test_train + "\class1\\" + image + ".png", tempImg)
                    if label == 0:
                        cv2.imwrite(path_test_train + "\class0\\" + image + ".png", tempImg)
                except:
                    print('Error Occured while processing images', imagePath)
                    pass


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=100, kernel_size=3,stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=100, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 50 x 50

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 25 x 25

            nn.Flatten(),
            nn.Linear(160000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3))

    def forward(self, x):
        return self.cnn_layers(x)


def trainModel(net, trainloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

    for epoch in range(10):
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            outputs = net.forward(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # TEST ACCURACY
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # END
            gc.collect()
            torch.cuda.empty_cache()
        #print('Accuracy of the network on Epoch', epoch, ': %d %%' % (
         #       100 * correct / total))
    return net

def eval_model(net, testloader, save):
    correct = 0
    total = 0
    image = []
    pred_data = []
    test_data = []
    for i, data in enumerate(testloader, 0):
        images, labels = data
        # 32plt.plot(images.numpy())
        outputs = net(images)
        test_data.extend(labels.numpy().tolist())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        pred_data.extend(predicted.numpy().tolist())
        correct += (predicted == labels).sum().item()
        if save:
            sample_fname, _ = testloader.dataset.samples[i]
            image.append(sample_fname)

    print('Accuracy of the network on ', len(test_data), ' test images: %d %%' % (
            100 * correct / total))
    print("\n")
    return (image, test_data, pred_data)


def bais_test_male(net):
    print("---------------------------MALE BIAS REPORT---------------------------------\n")
    data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Resize((constants.imageSize, constants.imageSize))])
    Male_testset = torchvision.datasets.ImageFolder(root=TESTMALE, transform=data_transform)
    Male_testloader = torch.utils.data.DataLoader(Male_testset, batch_size=1, shuffle=True, num_workers=8)
    image, test_data, pred_data = eval_model(net, Male_testloader, True)
    data = {"images": image, "expected": test_data, "predicted": pred_data}
    pd.DataFrame(data, columns=["images", "expected", "predicted"]).to_csv(path_or_buf='Male.csv')
    metricsEvaluation.evaluateCNNModel(test_data, pred_data)

def bais_test_female(net):
    print("\n\n ---------------------------------FEMALE BIAS REPORT--------------------------\n\n")
    
   
    data_transform = transforms.Compose([ transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Resize((constants.imageSize, constants.imageSize))])

    
    Female_testset = torchvision.datasets.ImageFolder(root=TESTFEMALE, transform=data_transform)
    Female_testloader = torch.utils.data.DataLoader(Female_testset, batch_size=1, shuffle=True, num_workers=8)
    image, test_data, pred_data = eval_model(net, Female_testloader, True)
    data = {"images": image, "expected": test_data, "predicted": pred_data}
    pd.DataFrame(data, columns=["images", "expected", "predicted"]).to_csv(path_or_buf='Female.csv')
    metricsEvaluation.evaluateCNNModel(test_data, pred_data)

if __name__ == '__main__':
    #ImageDataSetPrep("test")
    data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                         transforms.ToTensor()])
    testset = torchvision.datasets.ImageFolder(root=constants.TEST, transform=data_transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=8)

    PATH = "trained_cnn_Model.pt"
    if not os.path.exists(PATH):
        ImageDataSetPrep("train")
        skf = StratifiedKFold(n_splits=10)
        trainset = torchvision.datasets.ImageFolder(root=constants.Train, transform=data_transform)
        for i, (train_index, valid_index) in enumerate(skf.split(trainset, trainset.targets)):
            net = Net()
            data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                 transforms.ToTensor()])
            train = torch.utils.data.Subset(trainset, train_index)
            test = torch.utils.data.Subset(trainset, valid_index)

            trainloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True, num_workers=0,
                                                      pin_memory=False)
            validloader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True, num_workers=0,
                                                      pin_memory=False)
            trainModel(net, trainloader)
            #Changes modified by Pushpa
            print("******Result Metrics for FOLD->",i,"******")
            test_d,pred_d = eval_model(net, validloader,False)
            metricsEvaluation.evaluateCNNModel(test_d, pred_d)
            
        # Save
        torch.save(net, PATH)
    else:
        print("Loading Existing Train Model from ....",PATH)
        net = torch.load(PATH)
    
    #bais_test_female(net)
    #bais_test_male(net)
    
    print("Evaluating the Test Data from ::",constants.TEST)
    image,test_data, pred_data = eval_model(net, testloader,True)
    metricsEvaluation.evaluateCNNModel(test_data, pred_data)
    data = {"images": image, "expected": test_data, "predicted": pred_data}
    pd.DataFrame(data, columns=["images", "expected", "predicted"]).to_csv(path_or_buf='output.csv')
   
   

    