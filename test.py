import numpy as np
from dataset.verse_loader import get_loader_verse
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.evaluation.metric import dice, hausdorff_distance_95, recall, fscore
import argparse
import yaml 
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
from unet.basic_unet_denose_spm import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
from spm.unet import CONFIGS as CONFIGS_ViT_seg
from medpy import metric
set_determinism(123)
import os
import SimpleITK as sitk
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
data_dir = "../../autodl-tmp/UniSeg/Upstream/nnUNet_preprocessed/Task037_VerSe20binary/nnUNetData_plans_v2.1_stage0"

max_epoch = 300
batch_size = 1
val_every = 10
device = "cuda:0"

number_modality = 1
number_targets = 1 

def calculate_iou(predicted_mask, true_mask):
    intersection = np.logical_and(predicted_mask, true_mask)
    union = np.logical_or(predicted_mask, true_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def compute_uncer(pred_out):
    pred_out = torch.sigmoid(pred_out)
    pred_out[pred_out < 0.001] = 0.001
    uncer_out = - pred_out * torch.log(pred_out)
    return uncer_out

class DiffUNet(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        self.config=config
        
        self.embed_model = BasicUNetEncoder(3, number_modality, number_targets, [64, 64, 128, 256, 512, 64])

        self.model = BasicUNetDe(3, number_modality + number_targets, number_targets, [64, 64, 128, 256, 512, 64], 
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),config=self.config)
   
        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [50]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(1000)
        
    def forward(self, image=None, x=None, pred_type=None, step=None, embedding=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embedding=embedding)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)

            uncer_step = 4
            sample_outputs = []
            for i in range(uncer_step):
                sample_outputs.append(self.sample_diffusion.ddim_sample_loop(self.model, (1, number_targets, 128, 160, 96), model_kwargs={"image": image, "embeddings": embeddings}))

            sample_return = torch.zeros((1, number_targets, 128, 160, 96))

            for index in range(10):
# 
                uncer_out = 0
                for i in range(uncer_step):
                    uncer_out += sample_outputs[i]["all_model_outputs"][index]
                uncer_out = uncer_out / uncer_step
                uncer = compute_uncer(uncer_out).cpu()

                w = torch.exp(torch.sigmoid(torch.tensor((index + 1) / 10)) * (1 - uncer))
              
                for i in range(uncer_step):
                    sample_return += w * sample_outputs[i]["all_samples"][index].cpu()
        

                
            return sample_return
        
ious=[]
rs=[]
pres=[]

class VerSeTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py",config=None):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[128, 160, 96],
                                        sw_batch_size=1,
                                        overlap=0.5)
        
        self.config=config
        self.model = DiffUNet(config=self.config)


    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]
        label = label.float()
        
        return image, label 

    def validation_step(self, batch):
        image, label = self.get_input(batch)    
        
        output = self.window_infer(image, self.model, pred_type="ddim_sample")
        output = torch.sigmoid(output)
        output = (output > 0.5).float().cpu()
        

        d, w, h = label.shape[2], label.shape[3], label.shape[4]

        output = torch.nn.functional.interpolate(output, mode="nearest", size=(d, w, h))
        output = output.numpy()
        
        target = label.cpu().numpy()
        image_=image.cpu().numpy()
        dices = []
        hd = []
        
        
        
        c = 1
        for i in range(0, c):
            pred = output[:, i]
            gt = target[:, i]
            
           
            
            iou=calculate_iou(pred,gt)
            ious.append(iou)
            
            r=metric.binary.recall(pred, gt)
            rs.append(r)
            
            pre=metric.binary.precision(pred, gt)
            pres.append(pre)
            
            print("iou",iou)
            print("召回率",r)
            print("准确率",pre)
            
            if pred.sum() > 0 and gt.sum()>0:
                dice = metric.binary.dc(pred, gt)
                hd95 = metric.binary.hd95(pred, gt)
            elif pred.sum() > 0 and gt.sum()==0:
                dice = 1
                hd95 = 0
            else:
                dice = 0
                hd95 = 0
                
            
            dices.append(dice)
            hd.append(hd95)
            
        all_m = []
        for d in dices:
            all_m.append(d)
        for h in hd:
            all_m.append(h)

        print(all_m)
        
    
        return all_m 
    
if __name__ == "__main__":
    
    
    vit_name='ViT-L_16'
    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    config_vit.batch_size = batch_size
    img_size=[128,160,96]
    vit_patches_size=16
    # number of patches
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size[0] / vit_patches_size), int(img_size[1] / vit_patches_size), int(img_size[2] / vit_patches_size))
    ###
    config_vit.n_patches = int(img_size[0] / vit_patches_size) * int(img_size[1] / vit_patches_size) * int(img_size[2] / vit_patches_size)
    config_vit.n_patches = int(img_size[0] / vit_patches_size) * int(img_size[1] / vit_patches_size) * int(img_size[2] / vit_patches_size)
    config_vit.h = int(img_size[0] / vit_patches_size)
    config_vit.w = int(img_size[1] / vit_patches_size)
    config_vit.l = int(img_size[2] / vit_patches_size)

    train_ds, val_ds,test_ds = get_loader_verse(data_dir=data_dir, list_dir='data_split1',batch_size=batch_size, fold=0)
    
    trainer = VerSeTrainer(env_type="pytorch",
                                    max_epochs=max_epoch,
                                    batch_size=batch_size,
                                    device=device,
                                    val_every=val_every,
                                    num_gpus=1,
                                    master_port=17751,
                                    training_script=__file__,
                                    config=config_vit)

    logdir = "./logs_verse/diffusion_seg_all_loss_embed/model5/best_model_0.9049.pt"
    trainer.load_state_dict(logdir)
    v_mean, _ = trainer.validation_single_gpu(val_dataset=test_ds)

    print(f"v_mean is {v_mean}")
    print(f"iou is {sum(ious)/73}")
    print(f"回归率 is {sum(rs)/73}")
    print(f"准确率 is {sum(pres)/73}")