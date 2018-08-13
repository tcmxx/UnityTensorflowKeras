# Use MAES Optimization to Find the Best Solution

Covariance Matrix Adaptation Evolution Strategy(CMA-ES) is a type of evolutionary algorithm. See [Wikipedia Page](https://en.wikipedia.org/wiki/CMA-ES) for details.

To use our codes, you just need to understand the overall process. 
1. Initialize the optimzier with initial guess, and get the first generation of children from the optimizer.
2. Tell the optimizer how good each child in this generation is by asisgn a value to each of them.
3. The optimizer will update its covariance matrix based on how good those children are, and generate the next generation of children.
4. Repeat 2 and 3 until the you are satisfied with the result, or until you give up.

We provided some helper script in Unity for you to use MAES optimization without doing too much coding, whether you are using Unity's ML-Agents or not.

If you want to use the low level optimizers directly, check `LMMAES`, `MAES` classes and `IMAES` interface.

## Use ESOptimizer.cs
Example scene: `UnityTensorflow/Examples/IntelligentPool/BilliardMAESOnly-OneShot-UseMAESDirectly`.

`ESOptimizer.cs` is a helper script that you can attach to a GameObject and use it easily. Here are the steps:
1. Attach a `ESOptimizer.cs` to any GameObject, and set the parameters in inspector as you want(See [MAES parameters](#maes-parameters) for their meaning).
2. Implement `IESOptimizable` interface for the AI agent you want to optmizer. 
```csharp
public interface IESOptimizable {

    /// <summary>
    /// Evaluate a batch of params. 
    /// </summary>
    /// <param name="param">Each item in the list is a set of parameters.</param>
    /// <returns>List of values of each parameter set in the input</returns>
    List<float> Evaluate(List<double[]> param);

    /// <summary>
    /// Return the dimension of the parameters
    /// </summary>
    /// <returns>dimension of the parameters</returns>
    int GetParamDimension();
}
```
Note that the `Evaluate` method above should be a batch operation. Each item in the input list is one child and you need to return the values of all children in the input list. 

3. Call one of the following two methods based on your need:
```csharp
    /// <summary>
    /// Start to optimize asynchronized. It is actaually not running in another thread, but running in Update() in each frame of your game.
    /// This way the optimization will not block your game.
    /// </summary>
    /// <param name="optimizeTarget">Target to optimize</param>
    /// <param name="onReady">Action to call when optmization is ready. THe input is the best solution found.</param>
    /// <param name="initialMean">initial mean guess.</param>
    public void StartOptimizingAsync(IESOptimizable optimizeTarget, Action<double[]> onReady = null, double[] initialMean = null)
```
 or
 
```csharp
    /// <summary>
    /// Optimize and return the solution immediately.
    /// </summary>
    /// <param name="optimizeTarget">Target to optimize</param>
    /// <param name="initialMean">initial mean guess.</param>
    /// <returns>The best solution found</returns>
    public double[] Optimize(IESOptimizable optimizeTarget,  double[] initialMean = null)
```


## Use MAESDecision for ML-Agents
Example scenes: Under `UnityTensorflow/Examples/IntelligentPool/BilliardSLAndMAES-xxxx`, .

We also have a `DecisionMAES` class which implements [AgentDependentDecision](AgentDependentDeicision.md) using MAES. If your agent has implemented `IESOptimizable`, you can just attach `DecisionMAES.cs` to your agent and use it for [PPO](Training-PPO.md) or [Supervised Learning](Training-SL.md).

## TrainerMAES.cs
This is deprecated. But you can still use it. Just use `AgentES` as base class instead of `Agent`, and use TranerMAES as the Trainer for the CoreBrainInternalTrainable. 

Example scene: `UnityTensorflow/Examples/IntelligentPool/BilliardMAESOnly-OneShot-UseTrainer`.

## ESOptimizer parameters
The explanation of paramters that you can change in inspecotr of ESOptimizer.cs
- `iterationPerUpdate`: When use asynchronized optimization, the number of generation per frame. Adjust this depending on your speed of evaluation.
- `populationSize`: Number of children in each generation.
- `optimizerType`: MAES or LMMAES(Limitted Memory MAES). For small parameter dimension, use MAES and for larger one use LMMAES.
- `initialStepSize`: Initial s variance of the children. 
- `mode`: Maximize or minimize the value.
- `maxIteration`: The optimizer will automatically stop if reaches the max iteration.
- `targetValue`: The optimizer will automatically stop if the best solution reaches the target value.
- `evalutaionBatchSize`: What is the max batch size when evaluting. Might speed up the evaluation if your `IESOptimizable`'s `Evaluate(List<double[]> param)` method performs better for batch evalutation.
