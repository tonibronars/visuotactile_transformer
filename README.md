# Visuotactile Transformers
Visuotactile transformers is our final project (me and @nehasunil) for Computer Vision, 6.8300 at MIT. We implement (1) visuotactile classification and (2) self-supervised representation learning for combining visuotactile data, and compare performance when using ResNets and ViT for the image encoders.

## Datasets

We generate collocated visual (depth images) and tactile (contact images) in simulation for four objects, and release our datasets here. To access the datasets, unzip the `training_data.zip` and `validation_data.zip` files. Then, theS datasets we used for training are under: 
 
`data/{pin_big_subset,aux_big,stud_big,usb_big}_ang/`

The datasets we used for validation are under:

`data/{pin_big_subset,aux_big,stud_big,usb_big}_test/.`

## Training the classification network

We provide code for training the classification network with ResNets and ViT encoders for visuotactile data, vision-only data, and tactile-only data.

To train with ResNet encoders and ViT encoders, respectively, run: 

`python -m visuotactile_transfomer.scripts.train_resnet_classification`

`python -m visuotactile_transfomer.scripts.train_classification`


Modify the flags at the beginning of the training script to specify vision-only, tactile-only, or visuotactile, and the name of the model to save. Models will be saved in the `models/` folder in the main directory.

To evaluate the trained models on the validation set, run: 

`python -m visuotactile_transfomer.scripts.eval_resnet_classification`

`python -m visuotactile_transfomer.scripts.eval_classification`

For ResNet and ViT encoders, respectively. Again, you will need to modify the flags at the top of the scripts to specify whether to evaluate with vision-only, tactile-only, or both, as well as the name of the model to evaluate. The validation accuracy will be printed, and the predicted and true classes saved to the `eval/` folder in the main directory, under the model name.

# Training the representaion learning network

We provide code for training the joint representation learning network with ResNets and ViT encoders for visuotactile data, vision-only data, and tactile-only data.

To train with ResNet encoders and ViT encoders, respectively, run: 

`python -m visuotactile_transfomer.scripts.train_resnet`

`python -m visuotactile_transfomer.scripts.train`


Modify the flags at the beginning of the training script to specify vision-only, tactile-only, or visuotactile, and the name of the model to save. Models will be saved in the `models/` folder in the main directory.

To evaluate the trained models on the validation set, run: 

`python -m visuotactile_transfomer.scripts.eval`

You will need to modify the flags at the top of the scripts to specify whether to evaluate with ResNet or ViT encoders, as well as the name of the model to evaluate. A t-SNE plot of the joint representation will be displayed, where points are colored by object class. Dark colored points represent tactile embeddings, while light colored points represent vision embeddings.