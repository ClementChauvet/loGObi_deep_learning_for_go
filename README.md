## loGObi 

loGObi is a model of deep learning trained to predict the best moves to play given a transform of the current board and past moves.
It is intended to be used in a Monte Carlo Tree Search like explained in the Alpha GO paper.

The chosen architecture coded in the mobilenet_v3.py file is the result of months of tries to find the best architecture.
The detailed explanation is present in the pdf of this repository.

This repository is intended to work with golois and data from Tristan Cazenave's website : https://www.lamsade.dauphine.fr/~cazenave/DeepLearningProject.html


### Overview

bash/init.sh is a script to download the data and library 

src/training.py is a set of function to launch, save and monitor the training of models. It will create a save folder and save the training data and model every 20 epochs
If the training has been stopped, calling the train function will restart from the latest checkpoint

src/mobilenet_v3 contains functions used to define a mobilenet. When directly executed will create a model and launch its training


### How to use

To train any model, you first have to initialise the data and libraries by executing bash.init.sh
You can then execute src/mobilenet_v3.py to launch or relaunch the training 
