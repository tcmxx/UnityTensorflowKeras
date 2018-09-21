
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif
using MLAgents;
using KerasSharp.Engine.Topology;
using System;

/// <summary>
/// actor critic network abstract class
/// You can implement this abstract class and use it as scripable object to plugin to a SupervisedLearningModel.
/// </summary>
public abstract class SupervisedLearningNetwork : UnityNetwork
{

    /// <summary>
    /// Implement this method for building your network for continuous action space. You should use the inputs tensors and arguments to build a keras neural network.
    /// This SupervisedLearningModel will create the input tensors for you and call this method to build everything.
    /// All input and output tensors should have the first dimension as batch dimension.
    /// It outputs (action, variance) tensors
    /// </summary>
    /// <param name="inVectorObservation">The input vector observation tensor.</param>
    /// <param name="inVisualObservation">The input visual observation tensor.</param>
    /// <param name="inMemery">Input memory tensor. Not used right now. It is always null.Just ignore it.</param>
    /// <param name="outActionSize">the desired output action size</param>
    /// <returns>output (action tensor, variance tensor). 
    /// If continuous action space, output should be the action itself. The variance tensor should be [dynamic,1] shape currently(one variance for all actions output), or can be null if your network does not support var.
    /// </returns>
    public abstract ValueTuple<Tensor, Tensor> BuildNetworkForContinuousActionSapce(Tensor inVectorObservation, List<Tensor> inVisualObservation, Tensor inMemery, int outActionSize);



    /// <summary>
    /// Implement this method for building your network for discrete action space. You should use the inputs tensors and arguments to build a keras neural network.
    /// This SupervisedLearningModel will create the input tensors for you and call this method to build everything.
    /// All input and output tensors should have the first dimension as batch dimension.
    /// </summary>
    /// <param name="inVectorObservation">The input vector observation tensor.</param>
    /// <param name="inVisualObservation">The input visual observation tensor.</param>
    /// <param name="inMemery">Input memory tensor. Not used right now. It is always null.Just ignore it.</param>
    /// <param name="outActionSizes">the desired output action sizes. Each element in the array is the action size for a branch</param>
    /// <returns>output a list contains the logits of all actoins for each branch 
    /// </returns>
    public abstract List<Tensor> BuildNetworkForDiscreteActionSpace(Tensor inVectorObs, List<Tensor> inVisualObs, Tensor inMemery,  int[] outActionSizes);




    /// <summary>
    /// Return all weight tensors. Used for training and saving/loading
    /// </summary>
    /// <returns>list of all weights</returns>
    public abstract List<Tensor> GetWeights();

}
