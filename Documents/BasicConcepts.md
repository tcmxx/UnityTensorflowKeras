# Features and Basic Concepts

## Features
This repo provides or plans to provide following tools for game AI and machine learning training within Unity. You don't need python to run everything(you can make a game of training neural network!) and it is fully compatible with Unity's ML-Agents.

### Core Functionalities
1. Training with Reinforcement Learning
	* Using PPO(Proximal Policy Optimization) algorithm
    * You can provide heruistic(already know better behaviours) to speed up the training.
    * PPO with Neural Evolution(work in progess)
    
2. MAES[(Matrix Adaptation Evolution Strategy)](https://en.wikipedia.org/wiki/CMA-ES)
	* A genetic algorithm to find the best solution.
    * Can be used with or without ML-Agents.
3. Training with Supervise Learning
	* It is called imitation learning in ML-Agents. Here it is called supervised learning or mimic(I am too lazy to unify the name). (basically you show the correct action under different circumstances and let the neural network remember it.)
    * The correct action data can be collected from human playing or others, such as MAES.

4. Neural Evolution(work in progess)
	* Evolve the neural network's weights using MAES instead of gradien descent.

### Other tools
1. GAN(Generative adversarial network)
	* Including [Traning with Prediction to Stableize](https://www.semanticscholar.org/paper/Stabilizing-Adversarial-Nets-With-Prediction-Yadav-Shah/ec25504486d8751e00e613ca6fa64b256e3581c8).
	* [Improved Training of Wasserstein GANs with gradient penalty](http://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf)(future plan).
2. Heat map rendering tool
3. [Artistic Style Transfer](https://arxiv.org/abs/1705.08086)(future plan)
4. In editor progress visualization tool(future plan)

## Concepts
You can fisrt go through the [vverview of Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/ML-Agents-Overview.md), without python related stuff.

Assume that you are somehow familiar with Unity ML-Agents, then following will be some brief explanation of concepts/key conponents that are used in this repo.
(To be added)
* Trainer
* Model
* MEAS Optimizer
* UnityNetwork
* Agent Dependent Decision

## Features Not Gonna Have
1. Curriculum Training
	* It is pretty easy to implement curriculum learning by yourself because there is not communication between programs.
2. Recurrent Neural Network
	* Keras Sharp currently does not have recurrent neural network because it is not in the c library of c Tensorflow. I might implement it in the future, if I have time and want to, or CNTK can be used as backend instead of Tensorflow.
3. Training on mobile device
	* Not unless tensorflow start to include training on mobile build, or CNTK can be used as backend and it support mobile device. However, you can still use inference on mobile device.
