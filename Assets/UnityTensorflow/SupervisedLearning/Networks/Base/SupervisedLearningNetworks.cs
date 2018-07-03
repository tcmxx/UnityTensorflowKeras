
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif
using MLAgents;
using KerasSharp.Engine.Topology;


/// <summary>
/// actor critic network abstract class
/// </summary>
public abstract class SupervisedLearningNetwork : ScriptableObject
{


    /// <summary>
    /// 
    /// </summary>
    /// <param name="inVectorstate"></param>
    /// <param name="inVisualState"></param>
    /// <param name="inMemery"></param>
    /// <param name="outActionSize"></param>
    /// <param name="actionSpace"></param>
    /// <returns>output action tensor. If continuous action space, output should be the action itself. If discrete action space, the output should be probabilities of each action(softmax).</returns>
    public abstract Tensor BuildNetwork(Tensor inVectorstate, List<Tensor> inVisualState, Tensor inMemery, int outActionSize, SpaceType actionSpace);

    public abstract List<Tensor> GetWeights();

    public virtual void OnInspector()
    {
#if UNITY_EDITOR
        Editor.CreateEditor(this).OnInspectorGUI();
#endif
    }
}
