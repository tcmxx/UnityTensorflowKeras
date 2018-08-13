# Use Neural Evolution to optimize Neural Network

Neural Evolution is a totally different algorithm from gradient descent to optimize a neural network. 
It uses [Evolutionary Algorithm](https://en.wikipedia.org/wiki/Evolutionary_algorithm) to find the best weights of the neural network. 

A [paper](https://arxiv.org/pdf/1712.06567.pdf) from Uber shows that [genetic algorithm (GA)](https://en.wikipedia.org/wiki/Genetic_algorithm) can
,a subtype of Evolutionary Algorithm, performs well on deep reinformcement learning problems as well. However, here we are not gonna use GA yet, but MAES instead since we already have it.

Note that evolutionary algorithms can be very efficient if you can fun it in parallel on a lot of computers, because the children of each generation 
can usually be evalutad independently. But again, we are not doing this neither yet.

## Overall Steps
The steps are similiar to using other training method. See the scene `UnityTensorflow/Examples/3DBall/3DBallNE` for a simple example.

1. Create a environment using ML-Agent API. See the [instruction from Unity](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Create-New.md)
2. Change the BrainType of your brain to `InternalTrainable` in inspector.
3. Create a Trainer
	1. Attach a `TrainerNeuralEvolution.cs` to any GameObject.
    2. Create a `TrainerParamsNeuralEvolution` scriptable object with proper parameters in your project and assign it to the Params field in `TrainerNeuralEvolution.cs`.
    3. Assign the Trainer to the `Trainer` field of your Brain.
3. Create a Model
	1. Any model that implements INeuralEvolutionModel can be used as a model for neural evolution trainer. Currently both `RLModelPPO.cs`
  and `SupervisedLearningModel.cs` have that. Attach a one of those two to any GameObject.
    2. Create a Network scriptable object in your project for your model and attach it to the model.(See [Training with Proximal Policy Optimization(PPO)](https://github.com/tcmxx/UnityTensorflowKeras/blob/tcmxx/docs/Documents/Training-PPO.md) if you don't know how)
    3. Assign the created Model to the `modelRef` field of in `TrainerNeuralEvolution.cs`
    
5. Play! 

## Explanation of fields in the inspector
### TrainerNeuralEvolution.cs
* `isTraining`: Toggle this to switch between training and inference mode. Note that in inference mode, the best solutions from all previous generations will be used.
* `parameters`: You need to assign this field with a TrainerParamsNeuralEvolution scriptable object. 
* `continueFromCheckpoint`: If true, when the game starts, the trainer will try to load the saved checkpoint file to resume previous training.
* `checkpointPath`: the path of the checkpoint, including the file name. Note that there are two checkpoint files for neural evolution. One is the neural network's data which can be loaded by the model directly. The other one is called `YourName_NEData.xxx`, which stores the data of current generation. 
* `steps`: Just to show you the current step of the training.
* `currentEvaluationIndex`: The index of the child to evaluate in current generation.
* `currentGeneration`: How many generations have been evolved.
* `parameterDimension`: The total number of parameters to optimize.

### TrainerParamsNeuralEvolution
* `learningRate`: Not used.
* `maxTotalSteps`: Max steps the trainer will be training.
* `saveModelInterval`: The trained model will be saved every this amount of steps. However, the checkpoint file for generation data is saved after each evaluation of childen.
* `optimizerType`: Which MAES algorithm to use. Should choose LMMAES if the parameter dimension is big(which is common for neural network).
* `populationSize`: How many childrens to evaluate for each generation.
* `mode`: Whether the optimizer should maximize or minimize the value.
* `initialStepSize`: Initial step size of the optimizer. 
* `timeHorizon`: How many steps a sample will run to evaluate it. 

