import numpy as np
from dataset.verse_loader import get_loader_verse
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice,hausdorff_distance_95
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
from unet.basic_unet_denose_spm import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
import argparse
from monai.losses.dice import DiceLoss
import yaml
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
from spm.unet import CONFIGS as CONFIGS_ViT_seg

set_determinism(123)
import os

data_dir = "../../autodl-tmp/UniSeg/Upstream/nnUNet_preprocessed/Task037_VerSe20binary/nnUNetData_plans_v2.1_stage0"
logdir = "./logs_verse/diffusion_seg_all_loss_embed/"

model_save_path = os.path.join(logdir, "model")

env = "DDP" # or env = "pytorch" if you only have one gpu.
max_epoch = 300
batch_size = 1
val_every = 10
num_gpus = 1
device = "cuda:0"

number_modality = 1
number_targets = 1

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


    def forward(self, image=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)

            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, number_targets, 128, 160, 96), model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out

class VerSeTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py",config=None):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.config=config
        self.window_infer = SlidingWindowInferer(roi_size=[128, 160, 96],
                                        sw_batch_size=1,
                                        overlap=0.25)
        self.model = DiffUNet(config=self.config)

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-3)
        self.ce = nn.CrossEntropyLoss() 
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                  warmup_epochs=30,
                                                  max_epochs=max_epochs)

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)

    def training_step(self, batch):
        image, label = self.get_input(batch)
        x_start = label

        x_start = (x_start) * 2 - 1
        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")
        pred_xstart = self.model(x=x_t, step=t, image=image, pred_type="denoise")
        
        loss_dice = self.dice_loss(pred_xstart, label)
        loss_bce = self.bce(pred_xstart, label)

        pred_xstart = torch.sigmoid(pred_xstart)
        loss_mse = self.mse(pred_xstart, label)

        loss = loss_dice + loss_bce + loss_mse
        
        self.log("train_loss", loss, step=self.epoch)
        self.log("dice_loss",loss_dice,step=self.epoch)
        self.log("bce_loss",loss_bce,step=self.epoch)
        self.log("mse_loss",loss_mse,step=self.epoch)
        return loss
 
    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]

        label = label.float()
        return image, label 

    def validation_step(self, batch):
        image, label = self.get_input(batch)    
        
        output = self.window_infer(image, self.model, pred_type="ddim_sample")

        output = torch.sigmoid(output)

        output = (output > 0.5).float().cpu().numpy()



        target = label.cpu().numpy()
        dices = []
        hd = []

        pred_c = output[:, 0]
        target_c = target[:, 0]

        dices.append(dice(pred_c, target_c))
        hd.append(hausdorff_distance_95(pred_c, target_c))

        
        return dices,hd

    def validation_end(self, mean_val_outputs):
        dices,hd= mean_val_outputs
        mean_dice = sum(dices) / len(dices)
        mean_hd=sum(hd)/len(hd)
        
        mean_dice_value= mean_dice*100
        
        
        
        self.log("mean_dice%",mean_dice_value,step=self.epoch)
                 
        self.log("mean_dice", mean_dice, step=self.epoch)
        
        self.log("hd", hd, step=self.epoch)
        
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{mean_dice:.4f}.pt"), 
                                            delete_symbol="best_model")
        
        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{mean_dice:.4f}.pt"), 
                                        delete_symbol="final_model")

        print(f" mean_dice is {mean_dice}")
        

if __name__ == "__main__":
    train_ds, val_ds,test_ds = get_loader_verse(data_dir=data_dir, list_dir='data_split',batch_size=batch_size, fold=0)
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
    
    trainer = VerSeTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17751,
                            training_script=__file__,
                            config=config_vit)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
