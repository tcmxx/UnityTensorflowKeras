# Features and Basic Concepts

## Features
This repo provides or plans to provide following tools for game AI and machine learning training within Unity. You don't need python to run everything(you can make a game of training neural network!) and it is fully compatible with Unity's ML-Agents.

### Core Functionalities
1. Training with Reinforcement Learning
	* Using PPO(Proximal Policy Optimization) algorithm
    
2. CMA-ES[(Covariance Matrix Adaptation - Evolution Strategy)](https://en.wikipedia.org/wiki/CMA-ES)
	* A genetic algorithm to find the best solution.
    * Can be used with or without ML-Agents.
3. Training with Supervise Learning
	* It is called imitation learning in ML-Agents. Here it is called supervised learning or mimic(I am too lazy to unify the name). (basically you show the correct action under different circumstances and let the neural network remember it.)
    * The correct action data can be collected from human playing or others, such as CMAES.

4. Neural Evolution(Not gonna have for now)
	* Evolve the neural network's weights using MAES instead of gradien descent.
	* Currently only works for not very deep neural network.
### Other tools
1. GAN(Generative adversarial network)
	* Including [Traning with Prediction to Stableize](https://www.semanticscholar.org/paper/Stabilizing-Adversarial-Nets-With-Prediction-Yadav-Shah/ec25504486d8751e00e613ca6fa64b256e3581c8).
	* [Improved Training of Wasserstein GANs with gradient penalty](http://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf)(Not gonna have for now).
2. Heat map rendering tool
3. [Artistic Style Transfer](https://arxiv.org/abs/1705.08086)(Not gonna have for now)
4. In editor progress visualization tool(Not gonna have for now)
### Features Not Have
1. Curriculum Training
	* It is pretty easy to implement curriculum learning by yourself because there is not communication between programs.
2. Recurrent Neural Network
	* Keras Sharp currently does not have recurrent neural network because it is not in the c library of c Tensorflow. I might implement it in the future, if I have time and want to, or CNTK can be used as backend instead of Tensorflow.
3. Training on mobile device
	* Not unless tensorflow start to include training on mobile build, or CNTK can be used as backend and it support mobile device.
However, you can still use inference on mobile device.
4. Curiosity Module. 
	* I don't have time to implement it yet.
	
## Concepts
You don't have to read the folowing if you just want to use this repository as it is.

You can fisrt go through the [Overview of Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/ML-Agents-Overview.md), without python related stuff.

Assume that you are somehow familiar with Unity ML-Agents, then following will be some brief explanation of concepts/key conponents that are used in this repo.
(To be added)
### Trainer
The Brain in ML-Agent will communiate the with a Trainer to train the Model. We added a `CoreBrainInternalTrainable` on top of the existing core brains in ML-Agent which can communicate with our Trainers. The CoreBrainInternalTrainable works with any Monobehaviour that implement the `ITrainer` interface. 

We made some Trainers for you already for specific algorithm including PPO, SupervisedLearning and Evolution Strategy.

### Model
Models are the core of our AI. You can query information including the actions giving it the observations. Also, it provides interface to train the neural network.

Trainers will ask for actions and other training related data from Models during the training, and also ask to train the neural network when enough data can be provided.

### UnityNetwork

We defined some UnityNetwork scriptable objects, where you can easily define a neural network architecture for different Models, and use them as plugin modules(thanks to Unity's Scriptable Object). 

The models implemented by us usually need a network scriptable object that implement certain interface. We have already made the simple version of those network for you. However, you can also easily make your own customized network.



