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


## Installation
1. Clone the Unity MLAgent repo: https://github.com/Unity-Technologies/ml-agents. Or make sure you already have a project with ml-agent integrated.

2. Import the TenfowflowSharp plugin. One provided by Unity: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Using-TensorFlow-Sharp-in-Unity.md. It is not fully updated, and some operations are not supported. Or you can go to https://github.com/tcmxx/keras-sharp to download the version provided by me.

3. Clone this repo with submodules, for example you can use: 
 ```
 git clone --recurse-submodules https://github.com/tcmxx/UnityTensorflowKeras.git
 ```
   Note that you have to clone with submodules since it uses [KerasSharp](https://github.com/tcmxx/keras-sharp). You can also clone KerasSharp by yourself.
 
4. Copy the /UnityTensorflow folder under UnityTensorflowKeras's /Assets folder and put it into your Unity MLAgent's /Assets folder. Then you need to either replace the Brain.cs in Unity MLAgent's Assets/ML-Agents/Scripts folder with the one from UnityTensorflowKeras, or modify Brain.cs by youself to keep it up with the correct version: Add one line to the BrainType enum:
```    
    public enum BrainType
    {
        Player,
        Heuristic,
        External,
        Internal,
        InternalTrainable
    }
```
This enable you to select InternalTrainable as external brain type.

5. Delete the System.ValueTuple.dll in Unity MLAgent's /Assets/ML-Agents/Plugins folder if it is still there and you have updated Unity with .net 4.71 support. 
6. Done!

## Platforms:
Windows is almost fully supported. If you want to use GPU, CUDA and cuDNN are needed(See above). Mac should be fully supported if I have a Mac to build, but now it does not have Concat Gradient. Mac does not support GPU. Linux is not tested at all.

Android does not support any type of gradient/training. IOS is not tested a all.

## Documentation:
More documentation will be provided soon.
