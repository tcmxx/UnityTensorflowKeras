# UnityTensorflowKeras

- It is an extension of [Unity ML agent](https://github.com/Unity-Technologies/ml-agents) for deep learning, primarily reinforcement learning, with in-editor/in-game training support. It also provides interface for another optimization algorithm called MAES.

- It uses a modified version of [KerasSharp](https://github.com/tcmxx/keras-sharp) and [TensorflowSharp](https://github.com/migueldeicaza/TensorFlowSharp) as backend. No python is needed for model building/evaluation/training. You can even build a standalone with training capability.

- This repo is made for Aalto University's Computational Intelligence in Games course. [The original materials](https://github.com/tcmxx/CNTKUnityTools) are made with [CNTK](https://github.com/Microsoft/CNTK). Now it is remade with [Tensorflow](https://github.com/tensorflow/tensorflow). It will also be my master's thesis hopefully.

- Note: This project is still in development. Don't use this unless you know that you have time to check the sourcecodes!

## Features:
* Use your already made Unity ML-Agent, but enable learning in Unity editor/build without python.
* Reinforcement learning(using PPO) and supervised learning.
* Evolution Strategy (using Matrix Adaption Evolution Strategy(MEAS))

## Documentation
For more information including installation and usage instructions, go to [Document](Documents/Readme.md).


## Platforms:
Windows is almost fully supported. If you want to use GPU, CUDA and cuDNN are needed(See above). Mac should be fully supported if I have a Mac to build, but now it does not have Concat Gradient. Mac does not support GPU. Linux is not tested at all.

Android does not support any type of gradient/training. IOS is not tested a all.

## Future Plan:
We plan to keep this repo updated with latest game related machine learning technologies for the course every year.
