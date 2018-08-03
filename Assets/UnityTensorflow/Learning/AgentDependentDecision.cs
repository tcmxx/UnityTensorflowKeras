using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public abstract class AgentDependentDecision : MonoBehaviour
{

    public bool useDecision = true;

    private void Awake()
    {
        var agent = GetComponent<Agent>();
        Debug.Assert(agent != null, "Please attach the decision to the Agent you want to make decision on!");
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="agent"></param>
    /// <param name="vectorObs"></param>
    /// <param name="visualObs"></param>
    /// <param name="heuristicAction"></param>
    /// <param name="isTraining"></param>
    /// /// <param name="otherInfomation"></param>
    /// <returns></returns>
    public abstract float[] Decide(Agent agent, List<float> vectorObs, List<Texture2D> visualObs, List<float> heuristicAction, List<float> heuristicVariance = null);
}