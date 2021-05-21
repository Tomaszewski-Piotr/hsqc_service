# Python program to read
# image using PIL module

# importing PIL
from PIL import Image
import numpy as np
import images as images
import torch
from torch.optim import SGD
from torchvision.models import resnet18
from torchvision import transforms
from torch.utils import data
import copy
import pandas as pd
import os
import neptune

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

max_epochs = 2000

token = os.getenv('NEPTUNE_API_TOKEN')
upload = False

if token:
    print('Uploading results')
    neptune.init(project_qualified_name='piotrt/hsqc', api_token=os.getenv('NEPTUNE_API_TOKEN'))
    neptune.create_experiment(name='linear')
    upload = True
else:
    print('NEPTUNE_API_TOKEN not specified in the shell, not uploading results')


print('Running on: ', dev)

data_files = images.get_data_files()
print(len(data_files))

#x_all = np.array(shape=(len(data_files), channels, target_height, target_width), dtype=np.float32)
x_all = np.ndarray(shape=(len(data_files), images.target_height, images.target_width, images.channels), dtype=np.uint8)
y_all = np.ndarray(shape=(len(data_files), 5),
                     dtype=np.float32)

i = 0
no_file = len(data_files)
for file in data_files:
    print(i,'/', no_file, ' File:',str(file))
    #extract data scaled down to 224x224
    x_all[i] = np.array(images.preprocess_image(file))
    #extract required output
    y_all[i] = np.array(images.extract_values(file)).astype(np.float32)
    i+=1

#rotate x_all to be in CHW format (channel first)
print(x_all.shape)
x_all = np.moveaxis(x_all, 3, 1)
print(x_all.shape)

#make pytorch tensors
x_t = torch.tensor(x_all, dtype=torch.float32)
y_t = torch.tensor(y_all, dtype=torch.float32)
normalizer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
x_t = normalizer(x_t)
#x_t = x_t.to(dev)
#y_t = y_t.to(dev)
#create net
resnet = resnet18(pretrained=False, num_classes=5)
resnet.to(torch.device(dev))

optimizer = SGD(resnet.parameters(), lr=0.001)

loss_fn = torch.nn.MSELoss()

dataset = data.TensorDataset(x_t, y_t)

# divide 80/10/10 for train/valid/test datasets
train_size = int(0.8 * len(dataset))
valid_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

#prepare for training in batches
train_loader = data.DataLoader(train_dataset, batch_size=32)
x_val, y_val = valid_dataset[:]
x_test, y_test = test_dataset[:]

#safe test result as a file
y_act_np = y_test.cpu().detach().numpy()
act_df = pd.DataFrame(y_act_np, columns=images.value_names)
act_df.to_csv('actual_linear.csv')
x_val = x_val.to(dev)
y_val = y_val.to(dev)
x_test = x_test.to(dev)
y_test = y_test.to(dev)
best_loss = -1
best_model = resnet.state_dict()
best_epoch = -1
test_loss = -1
for epoch in range(max_epochs):
    print(epoch)
    #go through the batches and train
    for (x_batch, y_batch) in train_loader:
     #   print('batch')
        x_batch = x_batch.to(dev)
        y_batch = y_batch.to(dev)
        resnet.train()
        y_hat = resnet(x_batch)
        loss = loss_fn(y_hat, y_batch)
     #   print(loss)
     #   print(y_hat)
     #   print(y_batch)preprocess_image
        if upload:
            neptune.log_metric('Train loss', loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # evaluate using the validation dataset
    with torch.no_grad():
        resnet.eval()
        y_hat = resnet(x_val)
        cur_loss = loss_fn(y_hat, y_val)
        if upload:
            neptune.log_metric('Validation loss', cur_loss)
        #print(cur_loss)
        if best_loss == -1 or cur_loss < best_loss:
            best_loss = cur_loss
            best_epoch = epoch
            print('BESTIS: ', best_loss)
            best_model = copy.deepcopy(resnet.state_dict())
            torch.save(resnet.state_dict(), 'model.pt')
            #print(best_model)
            resnet.eval()
            y_hat_test = resnet(x_test)
            test_loss = loss_fn(y_hat_test, y_test)
            if upload:
                neptune.log_metric('Test loss', test_loss)
            print('Loss on test:', test_loss, ' epoch:', epoch)
            y_pred_np = y_hat_test.cpu().detach().numpy()
            pred_df = pd.DataFrame(y_pred_np, columns=images.pred_names)
            pred_df.to_csv('prediction_linear.csv')

print('Best model found on epoch: ', str(best_epoch))
print('Loss for best model: ', best_loss)
print('Test loss for best model: ', test_loss)
if upload:
    neptune.log_artifact('actual_linear.csv')
    neptune.log_artifact('prediction_linear.csv')
    neptune.stop()