from dataset import MultiPartitioningClassifier, cuda_base, device_ids, scenes, num_epochs
import yaml
from argparse import Namespace
import torch
import argparse

from torchvision.utils import save_image

with open('../config/base_config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model_params = config["model_params"]
tmp_model = MultiPartitioningClassifier(hparams=Namespace(**model_params))

train_data_loader = tmp_model.train_dataloader()
val_data_loader = tmp_model.val_dataloader()
# Choose the first n_steps batches with 64 samples in each batch
n_steps = 10

import os
import pandas as pd
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,models
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, confusion_matrix
from torchsummary import summary
import random
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from torchsummary import summary
from transformers import ViTModel
import warnings
import math
import time
warnings.filterwarnings("ignore", message="numerical errors at iteration 0")

device = torch.device(cuda_base if torch.cuda.is_available() else 'cpu')
def ttt(tuples):
    return torch.stack(list(tuples), dim=0)
    
def topk_accuracy(target, output, k):
    topn = np.argsort(output, axis = 1)[:,-k:]
    return np.mean(np.array([1 if target[k] in topn[k] else 0 for k in range(len(topn))]))


def adjust_learning_rate(num_epochs, optimizer, loader, step):
    max_steps = num_epochs * len(loader)
    warmup_steps = 2 * len(loader) ### In originalBT repo, this constant is 10
    base_lr = 0.1
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * 0.2
    optimizer.param_groups[1]['lr'] = lr * 0.0048


num_classes_coarse = 3298
num_classes_middle = 7202
num_classes_fine = 12893
learning_rate = config["lr"]
num_scene = int(scenes)

class GeoClassification(nn.Module):

    def __init__(self):
        super(GeoClassification, self).__init__()
        
        self.vit = ViTModel.from_pretrained('facebook/deit-tiny-patch16-224', output_attentions=True)
        for param in self.vit.parameters():
            param.requires_grad=False

#         self.classifier_1 = nn.Linear(self.vit.config.hidden_size*2, num_classes_coarse)
#         self.classifier_2 = nn.Linear(self.vit.config.hidden_size*2, num_classes_middle)
#         self.classifier_3 = nn.Linear(self.vit.config.hidden_size*2, num_classes_fine)
#         self.classifier_4 = nn.Linear(self.vit.config.hidden_size*2, num_scene)

        self.classifier_1 = nn.Linear(self.vit.config.hidden_size*2, num_classes_coarse)
        self.classifier_2 = nn.Linear(self.vit.config.hidden_size*2, 3000)
        self.classifier_3 = nn.Linear(self.vit.config.hidden_size*2, 6000)
        self.classifier_4 = nn.Linear(self.vit.config.hidden_size*2, num_scene)
    
        self.classifier_22 = nn.Linear(3000, num_classes_middle)
        self.classifier_33 = nn.Linear(6000, num_classes_fine)
    
    def forward(self, rgb_image, rgb_image_seg):
        
        vit_outputs=self.vit(rgb_image)
        vit_outputs_seg=self.vit(rgb_image_seg)
        
        outputs = self.vit(rgb_image).last_hidden_state
        outputs = outputs[:,0,:]
        outputs_seg = self.vit(rgb_image_seg).last_hidden_state
        outputs_seg = outputs_seg[:,0,:]
        #print("is this exist?:",self.vit(rgb_image).attentions)
        
        outputs_att=vit_outputs.attentions[11][:,0,0,0]
        outputs_attention=(ttt(outputs_att).detach()).cpu().numpy()+1
        outputs_attention=torch.from_numpy(outputs_attention).unsqueeze(1)
        
        outputs_att_seg=vit_outputs_seg.attentions[11][:,0,0,0]
        outputs_seg_attention=(ttt(outputs_att_seg).detach()).cpu().numpy()+1
        outputs_seg_attention=torch.from_numpy(outputs_seg_attention).unsqueeze(1)
        outputs_attention = outputs_attention.to(device)
        outputs_seg_attention=outputs_seg_attention.to(device)
        outputs=outputs.mul(outputs_attention)
        outputs_seg=outputs_seg.mul(outputs_seg_attention)
        
     
       
        final_output=torch.cat([outputs, outputs_seg],1)
        
        
        logits_geocell_coarse = self.classifier_1(final_output)
        logits_geocell_middle = self.classifier_22(self.classifier_2(final_output))
        logits_geocell_fine = self.classifier_33(self.classifier_3(final_output))
        logits_scene = self.classifier_4(final_output)
        return logits_geocell_coarse, logits_geocell_middle, logits_geocell_fine, logits_scene


model = GeoClassification()     
model = model.to(device)

param_weights = []
param_biases = []
for param in model.parameters():
    if param.ndim == 1:
        param_biases.append(param)
    else:
        param_weights.append(param)
parameters = [{'params': param_weights}, {'params': param_biases}]

n_total_steps = len(train_data_loader)
model = nn.DataParallel(model, device_ids=device_ids)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(parameters, lr=0.0005,momentum=0.9, weight_decay = config["weight_decay"])
#scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.05, total_steps=num_epochs+1, steps_per_epoch=n_total_steps, #epochs=num_epochs+1)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [4, 8, 10, 12], gamma= 0.5)
#momentum = config['momentum'],
# step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = config["milestones"], gamma= config["gamma"])
#    # MultiStepLR
#    params:
#      gamma: 0.5
#      milestones: [4, 8, 12, 13, 14, 15]
#print(summary(model, (3, 224, 224)))

target_total_test = []
predicted_total_test = []
model_outputs_total_test = []

import warnings
warnings.filterwarnings("ignore")



batch_wise_loss = []
batch_wise_micro_f1 = []
batch_wise_macro_f1 = []
# epoch_wise_top_1_coarse_accuracy = []
# epoch_wise_top_10_coarse_accuracy = []
# epoch_wise_top_50_coarse_accuracy = []
# epoch_wise_top_100_coarse_accuracy = []
# epoch_wise_top_200_coarse_accuracy = []
# epoch_wise_top_300_coarse_accuracy = []
# epoch_wise_top_500_coarse_accuracy = []

# epoch_wise_top_1_middle_accuracy = []
# epoch_wise_top_10_middle_accuracy = []
# epoch_wise_top_50_middle_accuracy = []
# epoch_wise_top_100_middle_accuracy = []
# epoch_wise_top_200_middle_accuracy = []
# epoch_wise_top_300_middle_accuracy = []
# epoch_wise_top_500_middle_accuracy = []

# epoch_wise_top_1_fine_accuracy = []
# epoch_wise_top_10_fine_accuracy = []
# epoch_wise_top_50_fine_accuracy = []
# epoch_wise_top_100_fine_accuracy = []
# epoch_wise_top_200_fine_accuracy = []
# epoch_wise_top_300_fine_accuracy = []
# epoch_wise_top_500_fine_accuracy = []

epoch_wise_top_1_accuracy = []
epoch_wise_top_10_accuracy = []
epoch_wise_top_50_accuracy = []
epoch_wise_top_100_accuracy = []
epoch_wise_top_200_accuracy = []
epoch_wise_top_300_accuracy = []
epoch_wise_top_500_accuracy = []

start_time=time.time()
scaler = torch.cuda.amp.GradScaler()
for epoch in range(num_epochs):
    steps = len(train_data_loader) * epoch
    print(len(train_data_loader))
    for i, (rgb_image, rgb_image_seg, label, lat, lon, scene) in enumerate(train_data_loader):
        
        rgb_image = rgb_image.type(torch.float32).to(device)
        #print("rgb_image:",rgb_image.shape)
        #lat=lat.to(device)
        #lon=lon.to(device)
        rgb_image_seg = rgb_image_seg.type(torch.float32).to(device)
        #print("rgb_image_seg:",rgb_image_seg.shape)
        label_coarse = label[0].to(device)
        label_middle = label[1].to(device)
        label_fine = label[2].to(device)
        scene = scene.type(torch.long).to(device)

        #adjust_learning_rate(num_epochs, optimizer, train_data_loader, steps)
        optimizer.zero_grad()
        steps += 1
        
         # Forward pass
        model.train()
        batch_size, n_crops, c, h, w = rgb_image.size()
        

        with torch.cuda.amp.autocast():
            outputs_geocell_coarse, outputs_geocell_middle, outputs_geocell_fine, outputs_scene =  model(rgb_image.view(-1, c, h, w), rgb_image_seg.view(-1, c, h, w))
            outputs_geocell_coarse = outputs_geocell_coarse.view(batch_size, n_crops, -1).mean(1)
            outputs_geocell_middle = outputs_geocell_middle.view(batch_size, n_crops, -1).mean(1)
            outputs_geocell_fine = outputs_geocell_fine.view(batch_size, n_crops, -1).mean(1)
            outputs_scene = outputs_scene.view(batch_size, n_crops, -1).mean(1)
            #pred_lat,pred_lon=partitionings[2].get_lat_lng(output_geocell_coarse)
            
            #gcd_coarse=vectorized_gc_distance(,lat,lon)
            loss_geocell_coarse = criterion(outputs_geocell_coarse, label_coarse)
            loss_geocell_middle = criterion(outputs_geocell_middle, label_middle)
            loss_geocell_fine = criterion(outputs_geocell_fine, label_fine)
            loss_scene = criterion(outputs_scene, scene)

            loss = 0.5*loss_geocell_coarse + 0.3*loss_geocell_middle + 0.2*loss_geocell_fine + loss_scene
        
        # Backward and optimize

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}',"time consumed:",time.time()-start_time)
            #print("time consumed:",time.time()-start_time)
           
        #if (i+1) == n_steps:
            #break
    try:
        scheduler.step()
    except:
        print("steps:",steps)
        print("len(traindloader):",len(train_data_loader))
        print("batchsize:",batch_size)  
        
    target_total_test = []
    predicted_total_test = []
    model_outputs_total_test = []
    #target_total_coarse_test = []
    #predicted_total_coarse_test = []
    #model_outputs_total_coarse_test = []
    #target_total_middle_test = []
    #predicted_total_middle_test = []
    #model_outputs_total_middle_test = []
    #target_total_fine_test = []
    #predicted_total_fine_test = []
    #model_outputs_total_fine_test = []

    with torch.no_grad():
        
        n_correct = 0
        n_samples = 0
#         n_fine_samples=0
#         n_coarse_samples=0
#         n_middle_samples=0
#         n_coarse_correct=0
#         n_middle_correct=0
#         n_fine_correct=0

        for i, (rgb_image, rgb_image_seg, label, _, _, scene) in enumerate(val_data_loader):
            
            rgb_image = rgb_image.type(torch.float32).to(device)
            rgb_image_seg = rgb_image_seg.type(torch.float32).to(device)
            #lat=lat.to(device)
            #lon=lon.to(device)

            #label_fine = label[2].to(device)
            #label_coarse = label[0].to(device)
            label_middle = label[1].to(device)
            #label_fine = label[2].to(device)
            scene = scene.type(torch.long).to(device)

            # Forward pass
            model.eval()
            batch_size, n_crops, c, h, w = rgb_image.size()
            outputs_geocell_coarse, outputs_geocell_middle, outputs_geocell_fine, outputs_scene =  model(rgb_image.view(-1, c, h, w), rgb_image_seg.view(-1, c, h, w))
            outputs_geocell_coarse = outputs_geocell_coarse.view(batch_size, n_crops, -1).mean(1)
            outputs_geocell_middle = outputs_geocell_middle.view(batch_size, n_crops, -1).mean(1)
            outputs_geocell_fine = outputs_geocell_fine.view(batch_size, n_crops, -1).mean(1)
            outputs_scene = outputs_scene.view(batch_size, n_crops, -1).mean(1)
            #print(outputs)
            # max returns (value ,index)
            _, predicted = torch.max(outputs_geocell_middle.data, 1)
#             _, predicted_fine = torch.max(outputs_geocell_fine.data, 1)
#             _, predicted_middle = torch.max(outputs_geocell_middle.data, 1)
#             _, predicted_coarse = torch.max(outputs_geocell_coarse.data, 1)
            #print(label)
            #print(predicted)
            n_samples += label_middle.size(0)
            n_correct += (predicted == label_middle).sum().item()
#             n_fine_samples += label_fine.size(0)
#             n_fine_correct += (predicted_fine == label_fine).sum().item()
#             n_middle_samples += label_middle.size(0)
#             n_middle_correct += (predicted_middle == label_middle).sum().item()
#             n_coarse_samples += label_coarse.size(0)
#             n_coarse_correct += (predicted_coarse == label_coarse).sum().item()

            target_total_test.append(label_middle)
            predicted_total_test.append(predicted)
            model_outputs_total_test.append(outputs_geocell_middle)
            
            
#             target_total_middle_test.append(label_middle)
#             predicted_total_middle_test.append(predicted_middle)
#             model_outputs_total_middle_test.append(outputs_geocell_middle)
            
#             target_total_coarse_test.append(label_coarse)
#             predicted_total_coarse_test.append(predicted_coarse)
#             model_outputs_total_coarse_test.append(outputs_geocell_coarse)



            target_inter = [t.cpu().numpy() for t in target_total_test]
            predicted_inter = [t.cpu().numpy() for t in predicted_total_test]
            outputs_inter = [t.cpu().numpy() for t in model_outputs_total_test]
            target_inter =  np.stack(target_inter, axis=0).ravel()
            predicted_inter =  np.stack(predicted_inter, axis=0).ravel()
            outputs_inter = np.concatenate(outputs_inter, axis=0)
            #########################################################################fine
#             target_fine_inter = [t.cpu().numpy() for t in target_total_fine_test]
#             predicted_fine_inter = [t.cpu().numpy() for t in predicted_total_fine_test]
#             outputs_fine_inter = [t.cpu().numpy() for t in model_outputs_total_fine_test]
#             target_fine_inter =  np.stack(target_fine_inter, axis=0).ravel()
#             predicted_fine_inter =  np.stack(predicted_fine_inter, axis=0).ravel()
#             outputs_fine_inter = np.concatenate(outputs_fine_inter, axis=0)
#             ###############################################################################middle
#             target_middle_inter = [t.cpu().numpy() for t in target_total_middle_test]
#             predicted_middle_inter = [t.cpu().numpy() for t in predicted_total_middle_test]
#             outputs_middle_inter = [t.cpu().numpy() for t in model_outputs_total_middle_test]
#             target_middle_inter =  np.stack(target_middle_inter, axis=0).ravel()
#             predicted_middle_inter =  np.stack(predicted_middle_inter, axis=0).ravel()
#             outputs_middle_inter = np.concatenate(outputs_middle_inter, axis=0)
#             ########################################################################################coarse
#             target_coarse_inter = [t.cpu().numpy() for t in target_total_coarse_test]
#             predicted_coarse_inter = [t.cpu().numpy() for t in predicted_total_coarse_test]
#             outputs_coarse_inter = [t.cpu().numpy() for t in model_outputs_total_coarse_test]
#             target_coarse_inter =  np.stack(target_coarse_inter, axis=0).ravel()
#             predicted_coarse_inter =  np.stack(predicted_coarse_inter, axis=0).ravel()
#             outputs_coarse_inter = np.concatenate(outputs_coarse_inter, axis=0)
        
        current_top_1_accuracy = topk_accuracy(target_inter, outputs_inter, k=1)
        epoch_wise_top_1_accuracy.append(current_top_1_accuracy)
        current_top_10_accuracy = topk_accuracy(target_inter, outputs_inter, k=10)
        epoch_wise_top_10_accuracy.append(current_top_10_accuracy)
        current_top_50_accuracy = topk_accuracy(target_inter, outputs_inter, k=50)
        epoch_wise_top_50_accuracy.append(current_top_50_accuracy)
        current_top_100_accuracy = topk_accuracy(target_inter, outputs_inter, k=100)
        epoch_wise_top_100_accuracy.append(current_top_100_accuracy)
        current_top_200_accuracy = topk_accuracy(target_inter, outputs_inter, k=200)
        epoch_wise_top_200_accuracy.append(current_top_200_accuracy)
        current_top_300_accuracy = topk_accuracy(target_inter, outputs_inter, k=300)
        epoch_wise_top_300_accuracy.append(current_top_300_accuracy)
        current_top_500_accuracy = topk_accuracy(target_inter, outputs_inter, k=500)
        epoch_wise_top_500_accuracy.append(current_top_500_accuracy)
       
        print(f' Accuracy of the network on the test set after Epoch {epoch+1} is: {accuracy_score(target_inter, predicted_inter)}')
        print(f' Top 2 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=2)}')
        print(f' Top 5 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=5)}')
        print(f' Top 10 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=10)}')
        print(f' Top 50 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=50)}')
        print(f' Top 100 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=100)}')
        print(f' Top 200 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=200)}')
        print(f' Top 300 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=300)}')
        print(f' Top 500 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=500)}')
        
        print(f' Best Top_1_accuracy on test set till this epoch: {max(epoch_wise_top_1_accuracy)} Found in Epoch No: {epoch_wise_top_1_accuracy.index(max(epoch_wise_top_1_accuracy))+1}')
        print(f' Best Top_10_accuracy on test set till this epoch: {max(epoch_wise_top_10_accuracy)} Found in Epoch No: {epoch_wise_top_10_accuracy.index(max(epoch_wise_top_10_accuracy))+1}')
        print(f' Best Top_50_accuracy on test set till this epoch: {max(epoch_wise_top_50_accuracy)} Found in Epoch No: {epoch_wise_top_50_accuracy.index(max(epoch_wise_top_50_accuracy))+1}')
        print(f' Best Top_100_accuracy on test set till this epoch: {max(epoch_wise_top_100_accuracy)} Found in Epoch No: {epoch_wise_top_100_accuracy.index(max(epoch_wise_top_100_accuracy))+1}')
        print(f' Best Top_200_accuracy on test set till this epoch: {max(epoch_wise_top_200_accuracy)} Found in Epoch No: {epoch_wise_top_200_accuracy.index(max(epoch_wise_top_200_accuracy))+1}')
        print(f' Best Top_300_accuracy on test set till this epoch: {max(epoch_wise_top_300_accuracy)} Found in Epoch No: {epoch_wise_top_300_accuracy.index(max(epoch_wise_top_300_accuracy))+1}')
        print(f' Best Top_500_accuracy on test set till this epoch: {max(epoch_wise_top_500_accuracy)} Found in Epoch No: {epoch_wise_top_500_accuracy.index(max(epoch_wise_top_500_accuracy))+1}')
        print(f' Top_1_accuracy: {epoch_wise_top_1_accuracy}')
        
        ##############################################################################################
        
#         current_top_1_coarse_accuracy = topk_accuracy(target_coarse_inter, outputs_coarse_inter, k=1)
#         epoch_wise_top_1_coarse_accuracy.append(current_top_1_coarse_accuracy)
#         current_top_10_coarse_accuracy = topk_accuracy(target_coarse_inter, outputs_coarse_inter, k=10)
#         epoch_wise_top_10_coarse_accuracy.append(current_top_10_coarse_accuracy)
#         current_top_50_coarse_accuracy = topk_accuracy(target_coarse_inter, outputs_coarse_inter, k=50)
#         epoch_wise_top_50_coarse_accuracy.append(current_top_50_coarse_accuracy)
#         current_top_100_coarse_accuracy = topk_accuracy(target_coarse_inter, outputs_coarse_inter, k=100)
#         epoch_wise_top_100_coarse_accuracy.append(current_top_100_coarse_accuracy)
#         current_top_200_coarse_accuracy = topk_accuracy(target_coarse_inter, outputs_coarse_inter, k=200)
#         epoch_wise_top_200_coarse_accuracy.append(current_top_200_coarse_accuracy)
#         current_top_300_coarse_accuracy = topk_accuracy(target_coarse_inter, outputs_coarse_inter, k=300)
#         epoch_wise_top_300_coarse_accuracy.append(current_top_300_coarse_accuracy)
#         current_top_500_coarse_accuracy = topk_accuracy(target_coarse_inter, outputs_coarse_inter, k=500)
#         epoch_wise_top_500_coarse_accuracy.append(current_top_500_coarse_accuracy)
       
#         print(f' Accuracy of the network on the test set after Epoch {epoch+1} is: {accuracy_score(target_coarse_inter, predicted_coarse_inter)}')
#         print(f' Top 2 accuracy on the testing coarse_: {topk_accuracy(target_coarse_inter, outputs_coarse_inter, k=2)}')
#         print(f' Top 5 accuracy on the testing coarse_: {topk_accuracy(target_coarse_inter, outputs_coarse_inter, k=5)}')
#         print(f' Top 10 accuracy on the testing coarse_: {topk_accuracy(target_coarse_inter, outputs_coarse_inter, k=10)}')
#         print(f' Top 50 accuracy on the testing coarse_: {topk_accuracy(target_coarse_inter, outputs_coarse_inter, k=50)}')
#         print(f' Top 100 accuracy on the testing coarse_: {topk_accuracy(target_coarse_inter, outputs_coarse_inter, k=100)}')
#         print(f' Top 200 accuracy on the testing coarse_: {topk_accuracy(target_coarse_inter, outputs_coarse_inter, k=200)}')
#         print(f' Top 300 accuracy on the testing coarse_: {topk_accuracy(target_coarse_inter, outputs_coarse_inter, k=300)}')
#         print(f' Top 500 accuracy on the testing coarse_: {topk_accuracy(target_coarse_inter, outputs_coarse_inter, k=500)}')
        
#         print(f' Best Top_1_accuracy on test set till this epoch coarse_: {max(epoch_wise_top_1_coarse_accuracy)} Found in Epoch No: {epoch_wise_top_1_coarse_accuracy.index(max(epoch_wise_top_1_coarse_accuracy))+1}')
#         print(f' Best Top_10_accuracy on test set till this epoch coarse_: {max(epoch_wise_top_10_coarse_accuracy)} Found in Epoch No: {epoch_wise_top_10_coarse_accuracy.index(max(epoch_wise_top_10_coarse_accuracy))+1}')
#         print(f' Best Top_50_accuracy on test set till this epoch coarse_: {max(epoch_wise_top_50_coarse_accuracy)} Found in Epoch No: {epoch_wise_top_50_coarse_accuracy.index(max(epoch_wise_top_50_coarse_accuracy))+1}')
#         print(f' Best Top_100_accuracy on test set till this epoch coarse_: {max(epoch_wise_top_100_coarse_accuracy)} Found in Epoch No: {epoch_wise_top_100_coarse_accuracy.index(max(epoch_wise_top_100_coarse_accuracy))+1}')
#         print(f' Best Top_200_accuracy on test set till this epoch coarse_: {max(epoch_wise_top_200_coarse_accuracy)} Found in Epoch No: {epoch_wise_top_200_coarse_accuracy.index(max(epoch_wise_top_200_coarse_accuracy))+1}')
#         print(f' Best Top_300_accuracy on test set till this epoch coarse_: {max(epoch_wise_top_300_coarse_accuracy)} Found in Epoch No: {epoch_wise_top_300_coarse_accuracy.index(max(epoch_wise_top_300_coarse_accuracy))+1}')
#         print(f' Best Top_500_accuracy on test set till this epoch coarse_: {max(epoch_wise_top_500_coarse_accuracy)} Found in Epoch No: {epoch_wise_top_500_coarse_accuracy.index(max(epoch_wise_top_500_coarse_accuracy))+1}')
#         print(f' Top_1_accuracy: {epoch_wise_top_1_coarse_accuracy}')
        
        
#                 ##############################################################################################
        
#         current_top_1_middle_accuracy = topk_accuracy(target_middle_inter, outputs_middle_inter, k=1)
#         epoch_wise_top_1_middle_accuracy.append(current_top_1_middle_accuracy)
#         current_top_10_middle_accuracy = topk_accuracy(target_middle_inter, outputs_middle_inter, k=10)
#         epoch_wise_top_10_middle_accuracy.append(current_top_10_middle_accuracy)
#         current_top_50_middle_accuracy = topk_accuracy(target_middle_inter, outputs_middle_inter, k=50)
#         epoch_wise_top_50_middle_accuracy.append(current_top_50_middle_accuracy)
#         current_top_100_middle_accuracy = topk_accuracy(target_middle_inter, outputs_middle_inter, k=100)
#         epoch_wise_top_100_middle_accuracy.append(current_top_100_middle_accuracy)
#         current_top_200_middle_accuracy = topk_accuracy(target_middle_inter, outputs_middle_inter, k=200)
#         epoch_wise_top_200_middle_accuracy.append(current_top_200_middle_accuracy)
#         current_top_300_middle_accuracy = topk_accuracy(target_middle_inter, outputs_middle_inter, k=300)
#         epoch_wise_top_300_middle_accuracy.append(current_top_300_middle_accuracy)
#         current_top_500_middle_accuracy = topk_accuracy(target_middle_inter, outputs_middle_inter, k=500)
#         epoch_wise_top_500_middle_accuracy.append(current_top_500_middle_accuracy)
       
#         print(f' Accuracy of the network on the test set after Epoch {epoch+1} is: {accuracy_score(target_middle_inter, predicted_middle_inter)}')
#         print(f' Top 2 accuracy on the testing middle_: {topk_accuracy(target_middle_inter, outputs_middle_inter, k=2)}')
#         print(f' Top 5 accuracy on the testing middle_: {topk_accuracy(target_middle_inter, outputs_middle_inter, k=5)}')
#         print(f' Top 10 accuracy on the testing middle_: {topk_accuracy(target_middle_inter, outputs_middle_inter, k=10)}')
#         print(f' Top 50 accuracy on the testing middle_: {topk_accuracy(target_middle_inter, outputs_middle_inter, k=50)}')
#         print(f' Top 100 accuracy on the testing middle_: {topk_accuracy(target_middle_inter, outputs_middle_inter, k=100)}')
#         print(f' Top 200 accuracy on the testing middle_: {topk_accuracy(target_middle_inter, outputs_middle_inter, k=200)}')
#         print(f' Top 300 accuracy on the testing middle_: {topk_accuracy(target_middle_inter, outputs_middle_inter, k=300)}')
#         print(f' Top 500 accuracy on the testing middle_: {topk_accuracy(target_middle_inter, outputs_middle_inter, k=500)}')
        
#         print(f' Best Top_1_accuracy on test set till this epoch middle_: {max(epoch_wise_top_1_middle_accuracy)} Found in Epoch No: {epoch_wise_top_1_middle_accuracy.index(max(epoch_wise_top_1_middle_accuracy))+1}')
#         print(f' Best Top_10_accuracy on test set till this epoch middle_: {max(epoch_wise_top_10_middle_accuracy)} Found in Epoch No: {epoch_wise_top_10_middle_accuracy.index(max(epoch_wise_top_10_middle_accuracy))+1}')
#         print(f' Best Top_50_accuracy on test set till this epoch middle_: {max(epoch_wise_top_50_middle_accuracy)} Found in Epoch No: {epoch_wise_top_50_middle_accuracy.index(max(epoch_wise_top_50_middle_accuracy))+1}')
#         print(f' Best Top_100_accuracy on test set till this epoch middle_: {max(epoch_wise_top_100_middle_accuracy)} Found in Epoch No: {epoch_wise_top_100_middle_accuracy.index(max(epoch_wise_top_100_middle_accuracy))+1}')
#         print(f' Best Top_200_accuracy on test set till this epoch middle_: {max(epoch_wise_top_200_middle_accuracy)} Found in Epoch No: {epoch_wise_top_200_middle_accuracy.index(max(epoch_wise_top_200_middle_accuracy))+1}')
#         print(f' Best Top_300_accuracy on test set till this epoch middle_: {max(epoch_wise_top_300_middle_accuracy)} Found in Epoch No: {epoch_wise_top_300_middle_accuracy.index(max(epoch_wise_top_300_middle_accuracy))+1}')
#         print(f' Best Top_500_accuracy on test set till this epoch middle_: {max(epoch_wise_top_500_middle_accuracy)} Found in Epoch No: {epoch_wise_top_500_middle_accuracy.index(max(epoch_wise_top_500_middle_accuracy))+1}')
#         print(f' Top_1_accuracy: {epoch_wise_top_1_middle_middle_accuracy}')
        
        
        
#                 ##############################################################################################
        
#         current_top_1_fine_accuracy = topk_accuracy(target_fine_inter, outputs_fine_inter, k=1)
#         epoch_wise_top_1_fine_accuracy.append(current_top_1_fine_accuracy)
#         current_top_10_fine_accuracy = topk_accuracy(target_fine_inter, outputs_fine_inter, k=10)
#         epoch_wise_top_10_fine_accuracy.append(current_top_10_fine_accuracy)
#         current_top_50_fine_accuracy = topk_accuracy(target_fine_inter, outputs_fine_inter, k=50)
#         epoch_wise_top_50_fine_accuracy.append(current_top_50_fine_accuracy)
#         current_top_100_fine_accuracy = topk_accuracy(target_fine_inter, outputs_fine_inter, k=100)
#         epoch_wise_top_100_fine_accuracy.append(current_top_100_fine_accuracy)
#         current_top_200_fine_accuracy = topk_accuracy(target_fine_inter, outputs_fine_inter, k=200)
#         epoch_wise_top_200_fine_accuracy.append(current_top_200_fine_accuracy)
#         current_top_300_fine_accuracy = topk_accuracy(target_fine_inter, outputs_fine_inter, k=300)
#         epoch_wise_top_300_fine_accuracy.append(current_top_300_fine_accuracy)
#         current_top_500_fine_accuracy = topk_accuracy(target_fine_inter, outputs_fine_inter, k=500)
#         epoch_wise_top_500_fine_accuracy.append(current_top_500_fine_accuracy)
       
#         print(f' Accuracy of the network on the test set after Epoch {epoch+1} is: {accuracy_score(target_fine_inter, predicted_fine_inter)}')
#         print(f' Top 2 accuracy on the testing fine_: {topk_accuracy(target_fine_inter, outputs_fine_inter, k=2)}')
#         print(f' Top 5 accuracy on the testing fine_: {topk_accuracy(target_fine_inter, outputs_fine_inter, k=5)}')
#         print(f' Top 10 accuracy on the testing fine_: {topk_accuracy(target_fine_inter, outputs_fine_inter, k=10)}')
#         print(f' Top 50 accuracy on the testing fine_: {topk_accuracy(target_fine_inter, outputs_fine_inter, k=50)}')
#         print(f' Top 100 accuracy on the testing fine_: {topk_accuracy(target_fine_inter, outputs_fine_inter, k=100)}')
#         print(f' Top 200 accuracy on the testing fine_: {topk_accuracy(target_fine_inter, outputs_fine_inter, k=200)}')
#         print(f' Top 300 accuracy on the testing fine_: {topk_accuracy(target_fine_inter, outputs_fine_inter, k=300)}')
#         print(f' Top 500 accuracy on the testing fine_: {topk_accuracy(target_fine_inter, outputs_fine_inter, k=500)}')
        
#         print(f' Best Top_1_accuracy on test set till this epoch fine_: {max(epoch_wise_top_1_fine_accuracy)} Found in Epoch No: {epoch_wise_top_1_fine_accuracy.index(max(epoch_wise_top_1_fine_accuracy))+1}')
#         print(f' Best Top_10_accuracy on test set till this epoch fine_: {max(epoch_wise_top_10_fine_accuracy)} Found in Epoch No: {epoch_wise_top_10_fine_accuracy.index(max(epoch_wise_top_10_fine_accuracy))+1}')
#         print(f' Best Top_50_accuracy on test set till this epoch fine_: {max(epoch_wise_top_50_fine_accuracy)} Found in Epoch No: {epoch_wise_top_50_fine_accuracy.index(max(epoch_wise_top_50_fine_accuracy))+1}')
#         print(f' Best Top_100_accuracy on test set till this epoch fine_: {max(epoch_wise_top_100_fine_accuracy)} Found in Epoch No: {epoch_wise_top_100_fine_accuracy.index(max(epoch_wise_top_100_fine_accuracy))+1}')
#         print(f' Best Top_200_accuracy on test set till this epoch fine_: {max(epoch_wise_top_200_fine_accuracy)} Found in Epoch No: {epoch_wise_top_200_fine_accuracy.index(max(epoch_wise_top_200_fine_accuracy))+1}')
#         print(f' Best Top_300_accuracy on test set till this epoch fine_: {max(epoch_wise_top_300_fine_accuracy)} Found in Epoch No: {epoch_wise_top_300_fine_accuracy.index(max(epoch_wise_top_300_fine_accuracy))+1}')
#         print(f' Best Top_500_accuracy on test set till this epoch fine_: {max(epoch_wise_top_500_fine_accuracy)} Found in Epoch No: {epoch_wise_top_500_fine_accuracy.index(max(epoch_wise_top_500_fine_accuracy))+1}')
#         print(f' Top_1_accuracy: {epoch_wise_top_1_fine_fine_accuracy}')
        
        
        print(f' Top_500_accuracy: {epoch_wise_top_500_accuracy}')

        if not os.path.exists('../saved_models'):
            os.makedirs('../saved_models')

        if(current_top_1_accuracy == max(epoch_wise_top_1_accuracy)):
#             torch.save({'Model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, #'../saved_models/ViT_RGB_FourTask.tar')
            torch.save(model, '../saved_models/ViT_RGB_FourTask.pt')
#'/home/sungkuk0620/Transformer_Based_Geo-localization/ViT_RGB_FourTask.tar'

print("======================================")
print("Training Completed, Evaluating the test set using the best model")
print("======================================")


#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

