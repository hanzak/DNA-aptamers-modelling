(Francais)

Le fichier principal utilisé pour entrainer les CNNs est : cnn_ml.ipynb
Ce fichier donne l'implémentation des fonctions ainsi que leur utilisation.

Pour que vous puissiez tester nos implémentations, nous avons créé un fichier dummy.
Le fichier dummy est : dummy_cnn.ipynb
Ce fichier dummy importe nos fonctions implémentées dans model_utils.py, ainsi que nos ensembles de données dummy.
Il est donc important, avant d'éxécuter le fichier dummy, d'avoir ces fichiers dans le même directory :
  - dummy_train.pkl
  - dummy_test.pkl
  - model_utils.py

Les modèles entrainés sont également fournis :
  - model_mfe.pth
  - model_hairpins.pth
  - model_struct.pth

Respectivement, pour MFE, hairpins et Structure secondaire.
Ces modèles ont été entrainés sur 5 millions de données. Les données sont : train_5M_struct.pkl dans le folder Data du main repos.
Les métriques présentées dans le rapport sont obtenus avec les données de data_test.pkl.

(Anglais)

The main file we used to train and implement our CNNs is : cnn_ml.ipynb
This main file gives all of our code with explanations on our implementations.

If you want to test our code, we created a separate dummy file : dummy_cnn.ipynb.
This dummy file uses the exact same logic as our main file but with the goal to see how the code works rather than to train the model.
We use very small datasets (length of 10) to train and test the models in our dummy file to give an overview of the functionalities of our code.

To test, you can launch all the cells in the notebook, make sure to have in the same folder :
  - dummy_train.pkl
  - dummy_test.pkl
  - model_utils.py

The trained models are also given in this folder : 
  - model_mfe.pth
  - model_hairpins.pth
  - model_struct.pth

These model are respectively for : MFE, hairpins and secondary structure. They were trained on 5 million sequences from the file : train_5M_struct.pkl (inside the Data folder in main repos).
The metrics shown in the rapport are from the model prediction on the dataset : data_test.pkl.