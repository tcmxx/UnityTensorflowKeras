# Training with Proximal Policy Optimization(PPO)

PPO is a popular reinforcement learning algorithm. See [this paper](https://arxiv.org/abs/1707.06347) for details.

Here, we are only going to tell how to use our existing code to train your ML-Agent environment in editor.

The example [Getting Started with the 3D Balance Ball Environment](Getting-Started-with-Balance-Ball.md) briefly shows how to use PPO to train an existing ML-Agent environment in editor. Here we are going to cover a little more details. 

## Overall Steps
1. Create a environment using ML-Agent API. See the [instruction from Unity](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Create-New.md)
3. Change the BrainType of your brain to `InternalTrainable` in inspector.
2. Create a Trainer
	1. Attach a `TrainerPPO.cs` to any GameObject.
    2. Create a `TrainerParamsPPO` scriptable object with proper parameters in your project and assign it to the Params field in `TrainerPPO.cs`.
    3. Assign the Trainer to the `Trainer` field of your Brain.
3. Create a Model
	1. Attach a `RLModelPPO.cs` to any GameObject.
    2. Create a `RLNetworkSimpleAC` scriptable with proper object in your project and assign it to the Network field in `RLModelPPO.cs`.
    3. Assign the created Model to the `modelRef` field of in `TrainerPPO.cs`
    
4. Play and see how it works.

## Explanation of fields in the inspector
We use similar parameters as in Unity ML-Agents. If something is confusing, read see their [document](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-PPO.md) for mode datails.

### TrainerPPO.cs
* `isTraining`: Toggle this to switch between training and inference mode. Note that if isTraining if false when the game starts, the training part of the PPO model will not be initialize and you won't be able to train it in this run. Also,
* `parameters`: You need to assign this field with a TrainerParamsPPO scriptable object. 
* `continueFromCheckpoint`: If true, when the game starts, the trainer will try to load the saved checkpoint file to resume previous training.
* `checkpointPath`: the path of the checkpoint, including the file name. 
* `steps`: Just to show you the current step of the training.

### TrainerParamsPPO
* `learningRate`: Learning rate used to train the neural network.
* `maxTotalSteps`: Max steps the trainer will be training.
* `saveModelInterval`: The trained model will be saved every this amount of steps.
* `rewardDiscountFactor`: Gamma. See PPO algorithm for details.
* `rewardGAEFactor`: Lambda. See PPO algorithm for details.
* `valueLossWeight`: Weight of the value loss compared with the policy loss in PPO.
* `timeHorizon`: Max steps when the PPO trainer will calculate the advantages using the collected data.
* `entropyLossWeight`: Weight of the entropy loss.
* `clipEpsilon`: See PPO algorithm for details. The default value is usually fine.
* `batchSize`: Mini batch size when training.
* `bufferSizeForTrain`: PPO will train the model once when the buffer size reaches this.
* `numEpochPerTrain`: For each training, the data in the buffer will be used repeatedly this amount of times.
* `useHeuristicChance`: See [Training with Heuristics](#training-with-heuristics).

### RLModelPPO.cs
* `checkpointToLoad`: If you assign a model's saved checkpoint file to it, this will be loaded when model is initialized, regardless of the trainer's loading. Might be used when you are not using a trainer.
* `Network`: You need to assign this field with a scriptable object that implements RLNetworkPPO.cs. 
* `optimizer`: The time of optimizer to use for this model when training. You can also set its parameters here.

### RLNetworkSimpleAC
This is a simple implementation of RLNetworkAC that you can create a plug it in as a neural network definition for any RLModelPPO. PPO uses actor/critic structure(See PPO algorithm).
- `actorHiddenLayers`/`criticHiddenLayers`: Hidden layers of the network. The array size if the number of hidden layers. In each element, there are for parameters that defines each layer. Those do not have default values, so you have to fill them.
	- size: Size of this hidden layer. 
    - initialScale: Initial scale of the weights. This might be important for training.Try something larger than 0 and smaller than 1.
    - useBias: Whether Use bias. Usually true.
    - activationFunction: Which activation function to use. Usually Relu.
- `actorOutputLayerInitialScale`/`criticOutputLayerInitialScale`/`visualEncoderInitialScale`: Initial scale of the weights of the output layers.
- `actorOutputLayerBias`/`criticOutputLayerBias`/`visualEncoderBias`: Whether use bias.

## Training with Heuristics
If you already know some policy that is better than random policy, you might give it as a hint to PPO to increase the training a bit. 

1. Implement the [AgentDependentDeicision](AgentDependentDeicision.md) for your policy and attach it to the agents that you want them to occasionally use this policy.
2. In your trainer parameters, set `useHeuristicChance` to larger than 0.
3. Use [TrainerParamOverride](TrainerParamOverride.md) to decrease the `useHeuristicChance` over time during the training.

Note that your AgentDependentDeicision is only used in training mode. The chance of using it in each step for agent with the script attached depends on `useHeuristicChance`.




