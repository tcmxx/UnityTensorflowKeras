using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public abstract class AgentDependentDecision : MonoBehaviour
{

    public bool useDecision = true;
    protected Agent agent;
    protected virtual void Awake()
    {
        agent = GetComponent<Agent>();
        Debug.Assert(agent != null, "Please attach the decision to the Agent you want to make decision on!");
    }
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
}