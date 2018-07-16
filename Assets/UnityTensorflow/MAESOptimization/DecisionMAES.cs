using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using MLAgents;
[RequireComponent(typeof(ESOptimizer))]
public class DecisionMAES : AgentDependentDecision
{
    protected ESOptimizer optimizer;
    
    public bool useMAESParamsFromAgent = true;
    private void Awake()
    {
        optimizer = GetComponent<ESOptimizer>();
    }

    public override float[] Decide(Agent agent, List<float> vectorObs, List<Texture2D> visualObs, List<float> heuristicAction )
    {

        Debug.Assert(agent is AgentES, "DesicionMAES required the agent to implement AgentES.");
        var agentES = agent as AgentES;

        if (useMAESParamsFromAgent)
        {
            optimizer.populationSize = agentES.populationSize;
            optimizer.targetValue = agentES.targetValue;
            optimizer.maxIteration = agentES.maxIteration;
            optimizer.initialStepSize = agentES.initialStepSize;
        }
        double[] best = optimizer.Optimize(agentES, null, heuristicAction.Select(t=>(double)t).ToArray());

        var result = Array.ConvertAll(best, t => (float)t);
        return result;
    }
    
}
