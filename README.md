# DiffUnet-SPM
DiffUnet-SPM is a model for 3D spine CT image segmentation



## Requirements

- Python3.8.10  
- Pytorch 1.8.1

'''
cd Diffunet-SPM
'''

  `pip install -r requirements.txt`



## Dataset

We use Verse2020 as our experimental dataset.

[VerSe`20: Large Scale Vertebrae Segmentation Challenge](https://verse2020.grand-challenge.org/)

[anjany/verse: Everything about the 'Large Scale Vertebrae Segmentation Challenge' @ MICCAI 2019-2020 (github.com)](https://github.com/anjany/verse)



### Preprocess

We use [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) for data preprocessing.



## Training

If you want to train the model from scratch:

`cd Diffunet-SPM/VerSe2020`

`python train.py`



### Testing

When the model is trained, please modify the corresponding paths in the file.

`python test.py`



### Thanks

Code copied a lot from [Diff-UNet](https://github.com/ge-xing/Diff-UNet/tree/main) and [Explicit-Shape-Priors](https://github.com/AlexYouXin/Explicit-Shape-Priors/tree/main).
