from dataset import MultiPartitioningClassifier, cuda_base, device_ids, scenes, num_epochs
import yaml
from argparse import Namespace
import torch
import argparse
import os
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
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
import logging
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
import matplotlib.pyplot as plt

with open('../config/base_config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
def topk_accuracy(target, output, k):
    topn = np.argsort(output, axis = 1)[:,-k:]
    return np.mean(np.array([1 if target[k] in topn[k] else 0 for k in range(len(topn))]))

model_params = config["model_params"]
#print(model_params)
tmp_model = MultiPartitioningClassifier(hparams=Namespace(**model_params))
val_data_loader = tmp_model.val_dataloader()
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

def ttt(tuples):
    return torch.stack(list(tuples), dim=0)
  
# def vectorized_gc_distance(latitudes, longitudes, latitudes_gt, longitudes_gt):
#     R = 6371
#     factor_rad = 0.01745329252
#     longitudes = factor_rad * longitudes
#     longitudes_gt = factor_rad * longitudes_gt
#     latitudes = factor_rad * latitudes
#     latitudes_gt = factor_rad * latitudes_gt
#     delta_long = longitudes_gt - longitudes
#     delta_lat = latitudes_gt - latitudes
#     subterm0 = torch.sin(delta_lat / 2) ** 2
#     subterm1 = torch.cos(latitudes) * torch.cos(latitudes_gt)
#     subterm2 = torch.sin(delta_long / 2) ** 2
#     subterm1 = subterm1 * subterm2
#     a = subterm0 + subterm1
#     c = 2 * torch.asin(torch.sqrt(a))
#     gcd = R * c
#     return gcd

# def gcd_threshold_eval(gc_dists, thresholds=[1, 25, 200, 750, 2500]):
#     # calculate accuracy for given gcd thresolds
#     results = {}
#     for thres in thresholds:
#         results[thres] = torch.true_divide(
#             torch.sum(gc_dists <= thres), len(gc_dists)
#         ).item()
#     return results

# Path="/home/sungkuk0620/GeoEstimation/resources/s2_cells/cells_50_1000.csv"
# class Partitioning:
#     def __init__(
#         self,
#         csv_file=Path,
#         shortname=None,
#         skiprows=None,
#         index_col="class_label",
#         col_class_label="hex_id",
#         col_latitute="latitude_mean",
#         col_longitude="longitude_mean",
#     ):

#         """
#         Required information in CSV:
#             - class_indexes from 0 to n
#             - respective class labels i.e. hexid
#             - latitude and longitude
#         """

#         logging.info(f"Loading partitioning from file: {csv_file}")
#         self._df = pd.read_csv(csv_file, index_col=index_col, skiprows=skiprows)
#         self._df = self._df.sort_index()

#         self._nclasses = len(self._df.index)
#         self._col_class_label = col_class_label
#         self._col_latitude = col_latitute
#         self._col_longitude = col_longitude

#         # map class label (hexid) to index
#         self._label2index = dict(
#             zip(self._df[self._col_class_label].tolist(), list(self._df.index))
#         )

#         self.name = csv_file  # filename without extension
#         if shortname:
#             self.shortname = shortname
#         else:
#             self.shortname = self.name

#     def __len__(self):
#         return self._nclasses

#     def __repr__(self):
#         return f"{self.name} short: {self.shortname} n: {self._nclasses}"

#     def get_class_label(self, idx):
#         return self._df.iloc[idx][self._col_class_label]

#     def get_lat_lng(self, idx):
#         x = self._df.iloc[idx]
#         return float(x[self._col_latitude]),float(x[self._col_longitude])

#     def contains(self, class_label):
#         if class_label in self._label2index:
#             return True
#         return False

#     def label2index(self, class_label):
#         try:
#             return self._label2index[class_label]
#         except KeyError as e:
#             raise KeyError(f"unkown label {class_label} in {self}")

# partitionings = []
# for shortname, path in zip(
#             ["coarse", "middle","fine"],
#             ["/home/sungkuk0620/GeoEstimation/resources/s2_cells/cells_50_1000.csv",
#              "/home/sungkuk0620/GeoEstimation/resources/s2_cells/cells_50_1000.csv",
#              "/home/sungkuk0620/GeoEstimation/resources/s2_cells/cells_50_1000.csv"]):
#     partitionings.append(Partitioning(path, shortname, skiprows=2))


# fine_gcds=torch.Tensor()
# middle_gcds=torch.Tensor()
# coarse_gcds=torch.Tensor()       
       

model = torch.load('/home/sungkuk0620/Transformer_Based_Geo-localization/saved_models/ViT_RGB_FourTask(9epoch die).pt')
#model.load_state_dict(saved_model['Model_state_dict'])
#optimizer.load_state_dict(saved_model['optimizer_state_dict'])
device = torch.device(cuda_base if torch.cuda.is_available() else 'cpu')
model.to(device)

target_total_test = []
predicted_total_test = []
model_outputs_total_test = []


with torch.no_grad():

        n_correct = 0
        n_samples = 0

        for i, (rgb_image, rgb_image_seg, label, lat, lon, scene) in enumerate(val_data_loader):

            rgb_image = rgb_image.type(torch.float32).to(device)

            #label_fine = label[2].to(device)
            label_coarse = label[0].to(device)
            label_middle = label[1].to(device)
            label_fine = label[2].to(device)
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
            #_, predicted = torch.max(outputs_geocell_fine.data, 1)
#             _, predicted_fine = torch.max(outputs_geocell_fine.data, 1)
#             _, predicted_middle = torch.max(outputs_geocell_middle.data, 1)
            _, predicted_fine = torch.max(outputs_geocell_fine.data, 1)
            
            #print("predicted_fine:",predicted_fine)
#             pred_lat_fine,pred_lon_fine=partitionings[2].get_lat_lng(predicted_fine[0].cpu())
#             pred_lats_fine = torch.tensor(pred_lat_fine, dtype=torch.float)
#             pred_lngs_fine = torch.tensor(pred_lon_fine, dtype=torch.float)
#             #a=vectorized_gc_distance(pred_lats_fine, pred_lngs_fine, lat, lon)
#             #빈 텐서 생성
#             fine_gcds=torch.cat([fine_gcds,vectorized_gc_distance(pred_lats_fine, pred_lngs_fine, lat, lon)])
            
            
#             #fine_result=gcd_threshold_eval(fine_gcds)
          
            
            
#             pred_lat_middle,pred_lon_middle=partitionings[1].get_lat_lng(predicted_middle[0].cpu())
#             pred_lats_middle = torch.tensor(pred_lat_middle, dtype=torch.float)
#             pred_lngs_middle = torch.tensor(pred_lon_middle, dtype=torch.float)
#             middle_gcds=torch.cat([middle_gcds,vectorized_gc_distance(pred_lats_middle, pred_lngs_middle, lat, lon)])
            
#             pred_lat_coarse,pred_lon_coarse=partitionings[0].get_lat_lng(predicted_coarse[0].cpu())
#             pred_lats_coarse = torch.tensor(pred_lat_coarse, dtype=torch.float)
#             pred_lngs_coarse = torch.tensor(pred_lon_coarse, dtype=torch.float)
#             coarse_gcds=torch.cat([coarse_gcds,vectorized_gc_distance(pred_lats_coarse, pred_lngs_coarse, lat, lon)])
            #print(label)
            #print(predicted)
            n_samples += label_fine.size(0)
            n_correct += (predicted_fine == label_fine).sum().item()
            import torchvision.transforms as T




              
            if n_correct==2:
                #np_arr = np.array(rgb_image[0,0,:].cpu(), dtype=np.uint8)
                save_image(rgb_image[0,0,:], 'image_name2.jpg')
#                 transform = T.ToPILImage()
#                 print(rgb_image.size())
#                 img = transform(rgb_image[0,0,:])
#                 imageseg=transform(rgb_image_seg[0,0,:])

#                 img.save("image_path4.jpg")
#                 imageseg.save("image_seg_path4.png")
                print(predicted_fine)
                print(rgb_image[0,0,:].size())
                print("lat, lon:",lat,lon)
                print('label:',label_fine)
               
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(to_pil_image((rgb_image[0,0,:])))
                plt.title('target')
               
                break

                
                

            target_total_test.append(label_fine)
            predicted_total_test.append(predicted_fine)
            model_outputs_total_test.append(outputs_geocell_fine)

            target_inter = [t.cpu().numpy() for t in target_total_test]
            predicted_inter = [t.cpu().numpy() for t in predicted_total_test]
            outputs_inter = [t.cpu().numpy() for t in model_outputs_total_test]
            #print(target_inter[-1].shape)
            #print(predicted_inter[-1].shape)
            #print(outputs_inter[-1].shape)
            target_inter =  np.stack(target_inter, axis=0).ravel()
            predicted_inter =  np.stack(predicted_inter, axis=0).ravel()
            outputs_inter = np.concatenate(outputs_inter, axis=0)
            
#         fine_gcd_result=gcd_threshold_eval(fine_gcds)
#         middle_gcd_result=gcd_threshold_eval(middle_gcds)
#         coarse_gcd_result=gcd_threshold_eval(coarse_gcds)
#         df=pd.DataFrame([coarse_gcd_result], index=['coarse'])
#         df2=pd.DataFrame([middle_gcd_result], index=['middle'])
#         df3=pd.DataFrame([fine_gcd_result], index=['fine'])
#         total_df=df.append(df2)
#         total_df=total_df.append(df3)
#         total_df.to_csv("/home/sungkuk0620/Transformer_Based_Geo-localization/resources/gcdresult.csv")
#         print(total_df)
#         print('-'*40)
        print(f' Accuracy of the network on the test set with the saved model is: {accuracy_score(target_inter, predicted_inter)}')
        print(f' Top 2 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=2)}')
        print(f' Top 5 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=5)}')
        print(f' Top 10 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=10)}')
        print(f' Top 50 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=50)}')
        print(f' Top 100 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=100)}')
        print(f' Top 200 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=200)}')
        print(f' Top 300 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=300)}')
        print(f' Top 500 accuracy on the testing: {topk_accuracy(target_inter, outputs_inter, k=500)}')

        
        
        
