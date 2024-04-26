# DNA-aptamers-modelling

Équipe :
David Arsenic (david.arsenic@umontreal.ca) 
Dean Johan Bell (dean.johan.bell@umontreal.ca) 
Haunui louis (haunui.louis@umontreal.ca) 
Hani Zaki (hani.zaki@umontreal.ca) 
Yehao Yan (yehao.yan@umontreal.ca)

## Description

Comparison of the performance of different machine learning models in the prediction of structual properties of DNA sequences. You will find the report in the `\Report` folder.

## Requirements

Git LFS is needed to access data.

## How to Run: Transformer
Python version: `3.9.5`

To run the code, follow these steps:

1. Install the dependencies using the following command: `pip install torch matplotlib tensorboard tqdm numpy scikit-learn`. 
2. Run the main.py file from the `DNA-aptamers-modelling` folder in the command line with the following arguments like so: `python Transfomer/packages/code/main.py <action> <model>`
  - `<action>` can take values of `train` or `evaluate`
    - If `train` is chosen, you will be asked to specify the datasize to use for training and the filename of the checkpoint file.
    - If `evaluate` is chosen, you will be asked to choose which datasize was used for training of the model you want to test, and then the filename of the model.pth file. You will then be asked if you want to test the model by intervals or not [y,n].
  - `<model>` can take values `encoder` (for Transformer with only an Encoder) or `decoder` (for Transformer Encoder & Decoder)
    
Model checkpoint will be saved in **Transformer/packages/model/model_checkpoint/**

Tensorboard data is saved in **Transformer/packages/model/runs/**

To view tensorboard data, run the following command: `tensorboard --logdir_spec=C:\path\to\runs\`

For the `\results` folder:
- The `\results` folder contains the results of the Transformer Encoder&Decoder model.
- `\results\final` contain the results on DNA sequences of length 10 to 50 from the data_test.pkl.
- `\results\gen` contain the results on DNA sequences of length 51 to 100 from the data_test.pkl.
- `\results\sim` contain the results on DNA sequences of length 10 to 50 from the data_test.pkl using the `generate_sequences_sim` method in the `transformer.py` file.


## How to Run: CNN
Python version: `3.9.5`

To run the code, follow the steps described in the README_CNN.txt inside of the CNN_main file.

## How to Run: LSTM
Install the required Python packages using pip: `python -m pip install -r LSTM/requirements.txt`.
