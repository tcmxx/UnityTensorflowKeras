# UnityTensorflowKeras

- It is an extension of [Unity ML agent](https://github.com/Unity-Technologies/ml-agents) for deep learning, primarily reinforcement learning, with in-editor/in-game training support. It also provides interface for another optimization algorithms such as MAES.

- It uses a modified version of [KerasSharp](https://github.com/tcmxx/keras-sharp) and [TensorflowSharp](https://github.com/migueldeicaza/TensorFlowSharp) as the backend, which usesTensorflow c++ lib. No python is needed for model building/evaluation/training. You can even build a standalone(an actual playable game!) with training capability.

- This repo is made for Aalto University's [Intellicent Computational Media](https://aaltoicmcourse.github.io/) course. The course includes two parts: Audio(Python) and Games(Unity and Python), and this repo contains the main materials for the Unity part of the course. This repo is a remake based on [the original materials](https://github.com/tcmxx/CNTKUnityTools), which are made with [CNTK](https://github.com/Microsoft/CNTK). It will also be part of my master's thesis.

- Note: This project is still in development. Don't use this unless you know that you have time to check the sourcecodes!


## Features:
* Use your already made Unity ML-Agent, but enable learning in Unity editor/build without python or extra coding.
* Reinforcement learning(PPO baseline) and supervised learning.
* Evolution Strategy (using Covariance Matrix Adaption Evolution Strategy(CMA-ES))
* Examples provided.

## Requirements: 
- [Unity ML agent v0.6](https://github.com/Unity-Technologies/ml-agents) 
- Unity 2018.1.6f1(Should be working with some of the newer versions as well).

## Platforms:
- Windows is almost fully supported. If you want to use GPU, CUDA and cuDNN are needed(Please google CUDA v9.0 and cudnn v7 and install them). 
- Mac should be fully supported if I have a Mac to build, but now it does not have Concat Gradient, which means the agent can not have both visual and vector observations at the same time, and discrete action branching is not supported neither. Mac does not support GPU. - - Linux is not tested at all.
- Android does not support any type of gradient/training. But you can use trained neural network on it. 
- IOS is not tested at all.
(Sorry that I am not a big fan of Apple products because they are expensive)

## Documentation
- Installation: https://github.com/tcmxx/UnityTensorflowKeras/blob/master/Documents/Installation.md
- For more information including installation and usage instructions, go to [Document](Documents/Readme.md).

## Future Plan:
We plan to keep this repo updated with latest game related machine learning technologies for the course every year.

Possible future plans/contributions:
* Updating [KerasSharp](https://github.com/tcmxx/keras-sharp), maybe with some basic recurrent NN.
* More example environments.
* Better API for in game usage and keep updated with Unity ML-Agents API.
* More algorithms including: Complete baseline PPO from ML-Agents(Curiosity Module and GAIL), Deep Q Learning, Deep Mimic, Evolved Policy Gradient, Genetic Algorithms and so on.
* Improving the logging tool.
* Graphic editor for neural network architecture

## License
[MIT](LICENSE).
