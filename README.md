# DNA-aptamers-modelling

Équipe :
David Arsenic (david.arsenic@umontreal.ca) 
Dean Johan Bell (dean.johan.bell@umontreal.ca) 
Haunui louis (haunui.louis@umontreal.ca) 
Hani Zaki (hani.zaki@umontreal.ca) 
Yehao Yan (yehao.yan@umontreal.ca)

## Description

Comparison of the performance of different machine learning models in the prediction of structual properties of DNA sequences.

## How to Run: Transformer
Python version: `3.9.5`

To run the code, follow these steps:

1. Install the dependencies using the following command: `pip install torch matplotlib tensorboard tqdm numpy sklearn`
2. Run the main.py file from the command line with the following arguments like so: `python path/to/main.py <action> <model>`
  - `<action>` can take values of `train` or `evaluate`
  - `<model>` can take values `encoder` (for Transformer with only an Encoder) or `decoder` (for Transformer Encoder & Decoder)
    - If `train` is chosen, you will be asked to specify the datasize to use for training and the filepath to save the checkpoint.
    - If `evaluate` is chosen, you will be asked to choose which datasize was used for training of the model you want to test, and then the filename of the model.pth file.

Model checkpoint will be saved in **Transformer/packages/model/model_checkpoint**

Tensorboard data is saved in **Transformer/packages/model/runs**

To view tensorboard data, run the following command: `tensorboard --logdir_spec=path\to\runs\`

