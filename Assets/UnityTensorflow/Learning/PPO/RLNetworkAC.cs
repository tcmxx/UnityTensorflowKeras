using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using KerasSharp.Engine.Topology;
#if UNITY_EDITOR
using UnityEditor;
#endif
using MLAgents;
/// <summary>
/// actor critic network abstract class
/// </summary>
public abstract class RLNetworkAC : UnityNetwork
{


    /// <summary>
    /// 
    /// </summary>
    /// <param name="inVectorstate"></param>
    /// <param name="inVisualState"></param>
    /// <param name="inMemery"></param>
    /// <param name="inPrevAction"></param>
    /// <param name="outActionSize"></param>
    /// <param name="actionSpace"></param>
    /// <param name="outAction">Output action. If action space is continuous, it is the mean; if aciton space is discrete, it is the probability of each action</param>
    /// <param name="outValue"></param>
    /// <param name="discreteActionProbabilitiesFor"></param>
    public abstract void BuildNetwork(Tensor inVectorstate, List<Tensor> inVisualState, Tensor inMemery, Tensor inPrevAction, int outActionSize, SpaceType actionSpace,
        out Tensor outAction, out Tensor outValue);

    public abstract List<Tensor> GetWeights();

}
