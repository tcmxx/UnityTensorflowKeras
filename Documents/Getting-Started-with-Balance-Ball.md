# Getting Started with the 3D Balance Ball Environment

This tutorial walks through the end-to-end process of converting the Unity ML-Agents' 3D Balance Ball environment into one that is trainable directly in Unity editor without coding extra stuff.

If you are not yet familiar with Unity's 3D Balance Ball Environment or you don't understant it at all, go to their [Understanding a Unity Environment (3D Balance Ball)](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Getting-Started-with-Balance-Ball.md#understanding-a-unity-environment-3d-balance-ball) section to understand what are Academy, Agent and Brain.

## Steps
### 1. Copy the scene

In the examples diretories provided by Unity ML-Agents, find the scene 3DBall and duplicate the scene.

![BallSceneDirectory](Images/Getting-Started-with-Balance-Ball/BallSceneDirectory.png)

### 2. Change the Brain Type to InternalTrainable
Go to the Ball3DBrain and change ites BrainType to Internal Trainable in inspector. If the Internal Trainable does not show up, make sure you follow this [installation step](https://github.com/tcmxx/UnityTensorflowKeras/blob/master/Documents/Installation.md#modify-braincs-to-add-support-for-training-inside-unity).

There will be a Trainer field showing up. You will create a Trainer in the next step and assign it to this field.

### 3. Create the Trainer and Trainer Parameters in the Scene
A trainer is something that handles the training process. It is based on the Unity ML-Agents' python script.

Add a new GameObject at any place, and attach a script called TrainerPPO.cs to it. Assign this script to the Trainer field in your Brain.
In the tutorial, we add it under Ball3DAcademy and name it Ball3DTrainer.

A trainer ususally needs a set of hyperparameters. Here for TrainerPPO, you need to create a scriptable object called TrainerParamsPPO and assign it to the Parameters field in TrainerPPO.

![CreateTrainerParams](Images/Getting-Started-with-Balance-Ball/CreateTrainerParams.png)

Finally check the Is Trainer field in the Trainer. The Trainer in the inspector should look like following. You can adjust the hyper parameters in it if you know what they mean.

![TrainerLookLike](Images/Getting-Started-with-Balance-Ball/TrainerWithParameters.png)

### 4. Create the Model and Network in the Scene
A model contains the high level interface to use the neural network for different learning algorithm. The trainer will communicate with the model to train the neural network and query actions. 

In our codes, most of models need a Network scripable object to define the detailed structure of the neural network, including how many hidden layers and so on. You can also easily write your own network. 


Add a new GameObject, attach a script called RLModelPPO.cs to it. Assign this script to the ModelRef field in your TrainerPPO.
Create two scriptable objects: TrainerParamsPPO and RLNetworkSimpleAC. Assign those to your TrainerPPO's Parameters fields and your RLModelPPO's network fields respectively.
Click Play and it will start!
You can click Window/Grapher from menu to monitor your training process.(It is a modified version of old Grapher, when it was still free. It seems to be not free nor opensource anymore...I will remove it if it is causing any problem.)
