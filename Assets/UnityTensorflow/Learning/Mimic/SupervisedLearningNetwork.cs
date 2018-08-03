
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
    /// Implement this method for building your network. You should use the inputs tensors and arguments to build a keras neural network.
    /// This SupervisedLearningModel will create the input tensors for you and call this method to build everything.
    /// All input and output tensors should have the first dimension as batch dimension.
    /// It outputs (action, variance) tensors
    /// </summary>
    /// <param name="inVectorObservation">The input vector observation tensor.</param>
    /// <param name="inVisualObservation">The input visual observation tensor.</param>
    /// <param name="inMemery">Input memory tensor. Not used right now. It is always null.Just ignore it.</param>
    /// <param name="outActionSize">the desired output action size</param>
    /// <param name="actionSpace">action space type</param>
    /// <returns>output (action tensor, variance tensor). 
    /// If continuous action space, output should be the action itself. The variance tensor should be [dynamic,1] shape currently(one variance for all actions output), or can be null if your network does not support var.
    /// If discrete action space, the output should be probabilities of each action(softmax). The variance tensor should be null.</returns>
    public abstract ValueTuple<Tensor,Tensor> BuildNetwork(Tensor inVectorObservation, List<Tensor> inVisualObservation, Tensor inMemery, int outActionSize, SpaceType actionSpace);

    /// <summary>
    /// Return all weight tensors. Used for training and saving/loading
    /// </summary>
    /// <returns>list of all weights</returns>
    public abstract List<Tensor> GetWeights();

}
