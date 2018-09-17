using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using KerasSharp.Engine.Topology;
#if UNITY_EDITOR
using UnityEditor;
#endif
using MLAgents;
/// <summary>
/// actor critic network abstract class. Inherit from this class if you want to build your own neural network structure for RLModePPO.
/// </summary>
public abstract class RLNetworkAC : UnityNetwork
{




    /// <summary>
    /// Impelment this abstract method to build your own neural network
    /// </summary>
    /// <param name="inVectorObs">input vector observation tensor</param>
    /// <param name="inVisualObs">input visual observation tensors</param>
    /// <param name="inMemery">input memory tensor. Not in use right now</param>
    /// <param name="inPrevAction">input previous action tensor. Noe in use right now</param>
    /// <param name="outActionSize">output action size. </param>
    /// <param name="outActionMean">outout value.</param>
    /// <param name="outValue">outout value.</param>
    /// <param name="outActionLogVariance">output outLogVariance. Only needed if the action space is continuous. It can either have batch dimension or not for RLModelPPO</param>
    public abstract void BuildNetworkForContinuousActionSapce(Tensor inVectorObs, List<Tensor> inVisualObs, Tensor inMemery, Tensor inPrevAction, int outActionSize,
        out Tensor outActionMean, out Tensor outValue, out Tensor outActionLogVariance);

    /// <param name="outActionSizes">output action sizes. Each element in the array is the size of each branch.(See unity ML agent for branches)/ </param>
    /// <param name="outActionLogits">Output action log probabilities.  Each element in the array is the probabilities of each branch.(See unity ML agent for branches)</param>
    public abstract void BuildNetworkForDiscreteActionSpace(Tensor inVectorObs, List<Tensor> inVisualObs, Tensor inMemery, Tensor inPrevAction,  int[] outActionSizes,
        out Tensor[] outActionLogits, out Tensor outValue);



    /// <summary>
    /// return all weights of the neural network
    /// </summary>
    /// <returns>List of tensors that are weights of the neural network</returns>
    public abstract List<Tensor> GetWeights();

    /// <summary>
    /// return all weights for the actor
    /// </summary>
    /// <returns>List of tensors that are weights used by the actor in the neural network</returns>
    public abstract List<Tensor> GetActorWeights();

    /// <summary>
    /// return all weights for the critic
    /// </summary>
    /// <returns>List of tensors that are weights used by the critic in the neural network</returns>
    public abstract List<Tensor> GetCriticWeights();
}
