# UnityTensorflowKeras

- It is an extension of [Unity ML agent](https://github.com/Unity-Technologies/ml-agents) for deep learning, primarily reinforcement learning, with in-editor/in-game training support. It also provides interface for another optimization algorithm called MAES.

- It uses a modified version of [KerasSharp](https://github.com/tcmxx/keras-sharp) and [TensorflowSharp](https://github.com/migueldeicaza/TensorFlowSharp) as backend. No python is needed for model building/evaluation/training. You can even build a standalone with training capability.

- This repo is made for Aalto University's Computational Intelligence in Games course. [The original materials](https://github.com/tcmxx/CNTKUnityTools) are made with [CNTK](https://github.com/Microsoft/CNTK). Now it is remade with [Tensorflow](https://github.com/tensorflow/tensorflow). It will also be my master's thesis hopefully.

- Note: This project is still in development. Don't use this unless you know that you have time to check the sourcecodes!

## Simple usage example:
Here is a simple example of how to use your existing UnityML agent and this repo to train neural netowkr using PPO in editor:
1. Copy you existing scene where you have implemented UnityML agent and Academy. Check [Making a new Learning environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Create-New.md) from UnityML's documentation.
2. You should have a Brain in your scene now. Change the BrainType to InternalTrainable. This option should be there if you have installed everything correctly.
3. Add a new GameObject, attach a script called TrainerPPO.cs to it. Assign this script to the Trainer field in your Brain.
4. Add a new GameObject, attach a script called RLModelPPO.cs to it. Assign this script to the ModelRef field in your TrainerPPO.
5. Create two scriptable objects: TrainerParamsPPO and RLNetworkSimpleAC. Assign those to your TrainerPPO's  Parameters fields and your RLModelPPO's network fields respectively.
6. Click Play and it will start!
7. You can click Window/Grapher from menu to monitor your training process.(It is a modified version of old Grapher, when it was still free. It seems to be not free nor opensource anymore...I will remove it if it is causing any problem.)


## Documentation
For more information including installation and usage instructions, go to [Document](Documents/Readme.md).


## Platforms:
Windows is almost fully supported. If you want to use GPU, CUDA and cuDNN are needed(See above). Mac should be fully supported if I have a Mac to build, but now it does not have Concat Gradient. Mac does not support GPU. Linux is not tested at all.

Android does not support any type of gradient/training. IOS is not tested a all.

## Documentation:
More documentation will be provided soon.
