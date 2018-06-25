using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(ESOptimizer))]
public class DecisionMAES : MonoBehaviour, IAgentDependentDecision
{
    protected ESOptimizer optimizer;

    private void Awake()
    {
        optimizer = GetComponent<ESOptimizer>();
    }

    public float[] Decide(List<float> vectorObs, List<Texture2D> visualObs, float reward, bool done, Agent agent)
    {
        Debug.Assert(agent is IESOptimizable, "DesicionMAES required the agent to implement AgentES.");
        double[] best = optimizer.Optimize(agent as IESOptimizable, null);

        var result = Array.ConvertAll(best, t => (float)t);
        return result;
    }
    
}
