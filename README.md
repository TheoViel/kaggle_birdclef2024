# 3rd Place Solution to the BirdCLEF 2024 Competition - Theo's part

**Authors :** [Theo Viel](https://github.com/TheoViel), [Jean-François Puget](https://github.com/jfpuget), [Christof Henkel](https://github.com/ChristofHenkel/)

This repo contains Theo's part of the solution to the 3rd place solution of the BirdCLEF 2024 challenge.
By itself, it achieves a public score of 0.72 ± 0.01 and a private score of 0.69 ± 0.01.

- Jean-François' part : https://github.com/jfpuget/birdclef-2024
- Christof's part: TODO

## Introduction - Adapted from [Kaggle](https://www.kaggle.com/competitions/birdclef-2024/discussion/511905)

Our pipeline is summarized below. A key ingredient of our solution is to use unlabeled soundscapes for pseudo labeling and model distillation. A number of models were trained with the training data, then used to predict labels on 5 second clips from unlabeled soundscapes. These were added to the original training data to train a new set of models used for the final submission. 

![](pipe.png)

### Data

Overall, we relied on knowledge acquired during previous competitions, and added some extra samples to fight class imbalance.

We use this year’s competition data plus the additional data from Xeno Canto shared in the forum, plus records from previous year competitions for the same species as this year. We capped the number of records per species to 500, keeping the most recent ones.

For each record we use a random crop of 5 seconds clip among the first 6 seconds, or optionally the last 6 seconds. For most of the models we used time shifting with a one second window as the only augmentation besides mixup. We use an additive mixup: mixed labels are the max of the labels of the two audios to be mixed.
We mostly use image models that take log mel spectrograms as input. For these we compute mel spectrograms with parameters chosen to have an image size of 224x224 .

### Models

#### First Level models

The cpu-only requirement was quite constraining for submissions, but this does not apply for pseudo-label generation, so we could use more backbones for first level models. When ensembling several models larger than the ones used at second level we perform what is known as model distillation. This is a rather powerful technique in general.

This pipeline uses a variety of CNNs (efficientnets, mobilenets, tinynets, mnasnets, mixnets) and Efficientvits ([b0, b1](https://arxiv.org/pdf/2205.14756), [m3](https://arxiv.org/pdf/2305.07027) trained on 224x224 log mel spectrograms.

#### Second Level models

Efficientvit-b0 showed great performances while still being very fast to infer. 5 folds take 40 minutes to submit using ONNX. We tried several models with similar throughput to Efficientvit-b0 and decided to also use an mnasnet-100 for diversity. For training second level models we added the unlabeled soundscapes with the predicted pseudo labels to the training data. This looks straight-forward but it took several attempts to find the correct way to do it. What worked fine was to use rather large batch sizes (128) and append a similar amount of pseudo labeled samples. 


## How to use the repository

### Prerequisites

- Clone the repository

- Download the data in the `input` folder:
  - [Competition data](https://www.kaggle.com/competitions/birdclef-2024/data)
  - [Extra xenocanto data](https://www.kaggle.com/datasets/ludovick/birdclef2024-additional-mp3)
  - [Other data](https://www.kaggle.com/datasets/theoviel/birdclef-2024-prev-data-fts)


The input folder should at least contain the following:
```
input
├── prev_comps_features       # From my dataset
├── train_audio               # From the competition data
├── unlabeled_soundscaptes    # From the competition data
├── xenocanto                 # From ludovick's dataset
│   ├── audio
│   └── BirdClef2024_additional.csv
├── df_extra_comp.csv         # From my dataset
└── train_metadata.csv        # From the competition data
```

- Setup the environment :
  - `pip install -r requirements.txt`

- I also provide trained model weights used to achieve private LB 0.70:
  - [Link](https://www.kaggle.com/datasets/theoviel/birdclef-2024-weights-3/).
  - The inference code is available [here](https://www.kaggle.com/code/theoviel/birdclef-2024-inf-v2)


### Run The pipeline

1. Preprocess the audios using `notebooks/Preparation.ipynb`
2. Train first level models using `bash train.sh`
3. Generate the associated pseudo-labels using `notebooks/Inference_v2.ipynb`
4. Train second level models using `bash train_2.sh`
    - Make sure to update the configs in `src/main_cnn_2.py` and `src/main_vit_2.py` to use the experiments from step 2.
5. Upload your weights to Kaggle, fork and run the [inference kernel](https://www.kaggle.com/code/theoviel/birdclef-2024-inf-v2)
    - Update the path to the models to your weights.

### Code structure

If you wish to dive into the code, the repository naming should be straight-forward. Each function is documented.
The structure is the following:

```
src
├── data
│   ├── dataset.py              # Dataset classes
│   ├── loader.py               # Dataloader
│   ├── mix.py                  # Mixup
│   ├── preparation.py          # Data preparation
│   └── processing.py           # Data processing 
├── inference                
│   └── predict.py              # Predict utils for PLs
├── model_zoo 
│   ├── layers.py               # Custom layers
│   ├── melspec.py              # Melspec and specaugment layers
│   └── models.py               # Classification model
├── training                        
│   ├── losses.py               # Losses
│   ├── main.py                 # k-fold and train function
│   ├── optim.py                # Optimizers
│   └── train.py                # Torch fit and eval functions
├── util
│   ├── logger.py               # Logging utils
│   ├── metrics.py              # Metrics for the competition
│   ├── plots.py                # Plotting utils
│   └── torch.py                # Torch utils
├── main_cnn_2.py               # 2nd level cnn traning
├── main_seg_cls.py             # 1st level cnn traning
├── main_seg.py                 # 2nd level vit traning
├── main.py                     # 1st level vit traning
└── params.py                   # Main parameters
``` 
