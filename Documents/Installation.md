# Installation

This repo uses source codes from two other repos:
* [Unity ML-Agent](https://github.com/Unity-Technologies/ml-agents). The interface of this repo is designed so that you can use any envirment designed for ML-Agents directly without recoding anything.
* [KerasSharp](https://github.com/tcmxx/keras-sharp). A modified version of KerasSharp is used to support Unity. It provides better interface for neural network, and also give the possibility to support CNTK in the future. KerasSharp is already included by this repo.

KerasSharp uses [TensorflowSharp](https://github.com/migueldeicaza/TensorFlowSharp) plugin built for Unity. 

## Install Unity.
Unity 2018 will probably work.

## Clone the Unity MLAgent
The repo in Github is at: https://github.com/Unity-Technologies/ml-agents. You can clone it anyway you want. All we need is the Assets/ML-Agents folder in its Unity project. All other python stuff is not needed.

If you already have a ML-Agents project with environment implemented, you can just skip this part.

Clone with git command:

    git clone https://github.com/Unity-Technologies/ml-agents.git

## Import the TenfowflowSharp Plugin
TensorflowSharp is necessary for running Tensorflow within Unity without python. 
There are two options for importing:
* One provided by Unity: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Using-TensorFlow-Sharp-in-Unity.md. It is not fully updated, and some operations are not supported. 
* Or you can go to [KerasSharp](https://github.com/tcmxx/keras-sharp) to download the version modified from Unity's plugin, with proper training support for some platforms. 

Notice that not all platforms are fully supported yet. Go to [KerasSharp](https://github.com/tcmxx/keras-sharp) to check all supported platforms.

## Clone this repo with submodules
Note that you have to clone this repo with its submodules. This repo uses KerasSharp as a submodule. 
 ```
 git clone --recurse-submodules https://github.com/tcmxx/UnityTensorflowKeras.git
 ```
If you don't what to clone with submodule, you can also clone the  [KerasSharp](https://github.com/tcmxx/keras-sharp) by yourself and copy the keras-sharp folder into your project.
 
## Copy the folders into your project.
You need to copy the necessary folders from this repo to your ML-Agents project:

It the /UnityTensorflow folder under UnityTensorflowKeras's /Assets folder, and you need to put it into your Unity MLAgent's /Assets folder. 

## Modify Brain.cs to Add Support For Training inside Unity

Finally,  you need to modify Brain.cs by a little so that there is a option to select InternalTrainable brain  as external brain type and enable training in Unity:

Add one line'InternalTrainable' to the BrainType enum:

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

## Test installation
Go to UnityTensorflow/Examples/3DBall/3DBall scene, and run it in editor. If no error message, then it is probably install correctly.

