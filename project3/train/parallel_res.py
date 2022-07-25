import csv
from sched import scheduler
from turtle import forward
from scipy import average
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset

#########
import os ##
import json ###
import cv2
import torch.optim
import torchvision.transforms as transforms
import sys
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
#from sklearn.model_selection import StratifiedKFold
import time
import numpy
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random


class MyDataset(Dataset) :
    def __init__(self,meta_path,root_dir,transform=None, is_train = True) :
        #meta_path: answer.json file path, root_dir: train/test data path
        super().__init__()
        self.meta_path = meta_path
        self.root_dir = root_dir
        self.transform = transform
        self.istrain = is_train ## use to distinguish different process b/w train, test
        #get a list of images
        self.filenames = os.listdir(self.root_dir)
#         with open(self.meta_path) as f:
#                 data =json.load(f)
#                 self.datas = data['annotations'] #long time consuming
    def __len__(self) :
        return len(self.filenames)

    def __getitem__(self,idx) : 
        if self.istrain: #train
            with open(self.meta_path) as f: #get meta data (filename, label)
                    data =json.load(f)
                    self.datas = data['annotations']
            filename = self.datas[idx]['file_name']
            label = int(self.datas[idx]['category'])
        else: #test
            filename = self.filenames[idx]
        imagepath = os.path.join(self.root_dir, filename)
        #get image using PIL
        image = Image.open(imagepath).convert('RGB')
        image = numpy.array(image)  
        if (self.transform): #apply transformation
          image = self.transform(image)
        if self.istrain:
            return image, filename, label
        else: #test: no label
            return image, filename

# classes for ResNet Model
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )
            
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class MyModel(nn.Module) : #ResNet50 architecture
    def __init__(self, block= BottleNeck, num_block=[3,4,6,3], num_classes=80, init_weights=True):
        super().__init__()
        self.in_channels=64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # weights inittialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0) 

def train(n_epochs, train_loader) :
# def train(n_epochs, train_loader, valid_loader, valstep) : #use when using kfold cross validation

    # val_step = valstep #use when using kfold cross validation
    # batch_size = batchsize
    # fold = ffold #use when using kfold cross validation
    # valid_dataloader = valid_loader #use when using kfold cross validation
    epochs = n_epochs #10000
    train_dataloader = train_loader

    #get current time -> folder name
    train_date = time.strftime("%y%m%d_%H%M", time.localtime(time.time()))
    # print(train_date)

    #log path for tensorboard
    log_path = os.path.join(os.getcwd(),"logs",train_date)
    print("log_path is {}".format(log_path))

    #model path to save '.pth'file
    model_path = os.path.join(os.getcwd(),"models",train_date)
    print("model_path is {}".format(model_path))
    if os.path.isdir(model_path) == False:
            os.makedirs(model_path)

    #writer for tensorboard
    writer = SummaryWriter(log_path)
    
    #update loss, f1score -> use when using kfold cross validation
    # best_loss = 9876543210.
    # best_f1score = 0

    # whether available GPU
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    #define model
    model = MyModel()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    #define loss funcion, optimizer, (scheduler)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(),lr=0.001, momentum=0.9,weight_decay=0.0005)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, 
    #                 momentum=0.875, weight_decay=3.05)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    print("start training")
    #Train
    for epoch in range(epochs):
        # list_predicts = [] #use when using kfold cross validation
        # list_labels = [] #use when using kfold cross validation
        
        model.train(True)
        len_trainbatch = len(train_dataloader)
        for i, data in enumerate(train_dataloader, 0):
            inputs, filename, labels = data
            # print(inputs)
            # print("inputs shape {}".format(inputs.shape))
            # print(labels)
            # print("labels shape {}".format(labels.shape))
            # print(filename)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            

            # print training log & write on tensorboard & reset vriables
            print(f'[Train] Epoch [{epoch+1}/{epochs}] | ' \
                    f'Batch [{i+1}/{len_trainbatch}] | ' \
                    f'total_loss:{loss.item():.5f} | ' \
                    )

            writer.add_scalar('train/loss', loss.item(), epoch*len_trainbatch+i)
            
        print("train loss of {}th epoch is {}".format(epoch,loss.item()))
        
        # #Validation -> use when using cross validation 

        # if epoch % val_step != 0:     
        #     continue  #Evaluation for every val_step
        # with torch.no_grad():
        #     model.eval()

        #     valid_loss = 0
        #     valid_f1score = 0
        #     valid_correct = 0
        #     valid_data_count = 0
        
        #     for i, data in enumerate(valid_dataloader):
        #         inputs, filename, labels = data
        #         inputs = inputs.to(device)
        #         labels = labels.to(device)
        #         outputs = model(inputs)
        #         loss = criterion(outputs, labels)

        #         # error metrics
        #         # output_cpu = outputs.detach().cpu().numpy()
        #         label_cpu = labels.detach().cpu().numpy()
        #         _, preds = torch.max(outputs, 1)
        #         preds_detach = preds.detach().cpu().numpy()
        #         list_labels.append(label_cpu)
        #         list_predicts.append(preds_detach)
        #         # print(output_cpu.shape)
        #         # f1score = f1_score(label_cpu, preds_detach, average = 'micro') #F1 score
        #         valid_loss += (loss.item() * len(inputs))
        #         valid_correct += preds.eq(labels.view_as(preds)).sum().item()
        #         # valid_f1score += (f1score.item() * len(inputs))
        #         valid_data_count += len(inputs)

        #     valid_loss /= valid_data_count
        #     total_labels = numpy.concatenate(list_labels)
        #     total_predicts = numpy.concatenate(list_predicts)
        #     valid_f1score = f1_score(total_labels,total_predicts, average = 'micro')
        #     valid_accuracy = 100.*valid_correct/valid_data_count
            
        #     #write validation log on tensorboard every val_step
        #     writer.add_scalar('validation/loss',valid_loss,epoch)
        #     writer.add_scalar('validation/f1score',valid_f1score,epoch)  
        #     writer.add_scalar('validation/accuracy',valid_accuracy,epoch)  

        # Save best validation loss, f1score model
        # if valid_loss < best_loss:
        #     best_loss = valid_loss
        #     best_net = model.state_dict()
        #     print(f'Best net Score : {best_loss:.4f}')
        #     torch.save(best_net, os.path.join(model_path, 'best_loss.pth'))
        # if valid_f1score > best_f1score:
        #     best_f1score = valid_f1score
        #     best_net = model.state_dict()
        #     print(f'Best f1score : {best_f1score:.4f}')
        #     torch.save(best_net, os.path.join(model_path, 'best_f1score.pth'))
            
        # Save every N(10) epoch
        save_epoch = 10
        if save_epoch > 0 and epoch % save_epoch == save_epoch-1:
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(model_path, str(epoch)+'.pth'))
        
    print('Finished Training')

    # # You SHOULD save your model by
    # # torch.save(model.state_dict(), './checkpoint.pth') 
    #save model after training
    torch.save(model.state_dict(), './model.pth') 


def get_model(model_name, checkpoint_path):
    
    model = model_name()
    model.load_state_dict(torch.load(checkpoint_path))
    
    return model

def test(batchsize):
    batch_size = batchsize
    # ######## not touch
    # model_name = MyModel
    # checkpoint_path = './model.pth' 
    # mode = 'test' 
    # data_dir = "./test_data"
    # meta_path = "./answer.json"
    # model = get_model(model_name,checkpoint_path)
    # ########

    data_transforms = {
    'train' : transforms.Compose([ #horizontal flip, random crop, normalization           
            transforms.ToPILImage(),
            transforms.Resize((256,256)),
            transforms.RandomCrop(224), #augmentation, input for ResNet
            transforms.RandomHorizontalFlip(p=0.5), #augmentation
            transforms.ToTensor(),
            transforms.Normalize([0.6071413, 0.5386342, 0.4528947],[0.22180064,0.24319556,0.26916447])
            # transforms.Normalize(train_data_mean, train_data_std) [0.6071413, 0.5386342, 0.4528947], [0.22180064,0.24319556,0.26916447]
        ]) , 
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            # transforms.RandomCrop(224), #not use during test
            transforms.ToTensor(),
            transforms.Normalize([0.6071413, 0.5386342, 0.4528947],[0.22180064,0.24319556,0.26916447])
            # transforms.Normalize(train_data_mean, train_data_std)
        ])
    }
    model_name = MyModel
    # checkpoint_path = './128.pth' 
    checkpoint_path = './models/220618_1754/49.pth' 
    # checkpoint_path = './models' 
    mode = 'test' 
    data_dir = "./val_data"
    meta_path = "./answer.json"
    model = get_model(model_name,checkpoint_path)
    
    # Create training and validation datasets
    test_datasets = MyDataset(meta_path, data_dir, data_transforms['test'],is_train= False) #self,meta_path,root_dir,transform=None, is_train = True

    # Create training and validation dataloaders
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False, num_workers=4)

    # Detect if we have a GPU available
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model = model.to(device)

    # Set model as evaluation mode
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    # Inference
    result = []
    for images, filename in tqdm(test_dataloader):
        num_image = images.shape[0]
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        for i in range(num_image):
            result.append({
                'filename': filename[i],
                'class': preds[i].item()
            })

    result = sorted(result,key=lambda x : int(x['filename'].split('.')[0]))
    
    #########
    # Save to csv
    with open('./result49.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['filename','class'])
        for res in result:
            writer.writerow([res['filename'], res['class']])
    #########


def main() :

    random.seed(53)

    # receive arguments: train, validation, test
    mode = sys.argv[1] #"train/test"
    print("mode is {}".format(mode))
    
    
    # # use only to calculate mean,std value 
    # train_metapath = os.path.join(os.getcwd(),'answer.json')
    # train_root_dir = os.path.join(os.getcwd(), 'train_data')
    # mean_transforms = transforms.Compose([
    #     transforms.ToTensor()
    # ])
    # first_dataset = MyDataset(train_metapath, train_root_dir, mean_transforms, is_train=True)

    # train_meanRGB = [numpy.mean(image.numpy(), axis=(1,2)) for image, filename, label in first_dataset ] #image, filename, label
    # train_meanR = numpy.mean([m[0] for m in train_meanRGB])
    # train_meanG = numpy.mean([m[1] for m in train_meanRGB])
    # train_meanB = numpy.mean([m[2] for m in train_meanRGB])

    # print("train_meanR",train_meanR)
    # print("train_meanG",train_meanG)
    # print("train_meanB",train_meanB)

    # train_stdRGB = [numpy.std(image.numpy(), axis=(1,2)) for image, filename, label in first_dataset ] #image, filename, label
    # train_stdR = numpy.mean([s[0] for s in train_stdRGB])
    # train_stdG = numpy.mean([s[1] for s in train_stdRGB])
    # train_stdB = numpy.mean([s[2] for s in train_stdRGB])

    # print("train_stdR",train_stdR)
    # print("train_stdG",train_stdG)
    # print("train_stdB",train_stdB)

    data_transforms = {
    'train' : transforms.Compose([ #horizontal flip, random crop, normalization           
            transforms.ToPILImage(),
            transforms.Resize((256,256)),
            transforms.RandomCrop(224), #augmentation, input for ResNet
            transforms.RandomHorizontalFlip(p=0.5), #augmentation
            transforms.ToTensor(),
            transforms.Normalize([0.6071413, 0.5386342, 0.4528947],[0.22180064,0.24319556,0.26916447])
            # transforms.Normalize(train_data_mean, train_data_std) [0.6071413, 0.5386342, 0.4528947], [0.22180064,0.24319556,0.26916447]
        ]) , 
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            # transforms.RandomCrop(224), #not use during test
            transforms.ToTensor(),
            transforms.Normalize([0.6071413, 0.5386342, 0.4528947],[0.22180064,0.24319556,0.26916447])
            # transforms.Normalize(train_data_mean, train_data_std)
        ])
    }

    if (mode == "train"):
        batch_size = 64
        #set data path (train)
        metapath = os.path.join(os.getcwd(),'answer.json')
        root_dir = os.path.join(os.getcwd(), 'train_data')
        dataset = MyDataset(metapath, root_dir, data_transforms['train'])
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        train(10000, trainloader) #epoch, trainloader
        ## K-fold 구현
        # kf = KFold(n_splits = 5)
        # batch_size = 64
        # valstep = 10
        # for fold, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
        #     train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        #     valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_idx)
        #     train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler = train_subsampler) #4->2
        #     valid_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler = valid_subsampler) #4->2
        #     train(10000, train_dataloader, valid_dataloader, valstep) #n_epochs, trainloader, validloader, valstep
    else: #test
        test(1) #batchsize

   

if __name__ == '__main__':
    main()
   