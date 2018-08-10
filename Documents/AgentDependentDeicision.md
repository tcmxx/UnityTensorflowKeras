# AgentDependentDeicision

If you want to use your own policy on a specific agent instead of using the Brain when using `TrainerPPO` or `TrainerMimic`, here is the way.

Implement the abstract class called AgentDependentDecision. You only need to implement one abstract method: 
```csharp
    /// <summary>
    /// Implement this method for your own ai decision.
    /// </summary>
    /// <param name="vectorObs">vector observations</param>
    /// <param name="visualObs">visual observations</param>
    /// <param name="heuristicAction">The default action from brain if you are not using the decision</param>
    /// <param name="heuristicVariance">The default action variance from brain if you are not using the decision. 
    /// It might be null if discrete aciton space is used or the Model does not support variance.</param>
    /// <returns>the actions</returns>
    public abstract float[] Decide(List<float> vectorObs, List<Texture2D> visualObs, List<float> heuristicAction, List<float> heuristicVariance = null);
```

Then, attach your new script to the agent you want to use your policy, and check the `useDecision` in inspector.

Note that your policy is only used under certain training setting when using `TrainerPPO` or `TrainerMimic`. See [Training with Proximal Policy Optimization(PPO)](Training-PPO.md) and [Training with Imitation(Supervised Learning)](Training-SupervisedLearning.md) for more details.
