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
    
    public bool useHeuristic = true;

    protected IESOptimizable optimizable = null;

    protected override void Awake()
    {
        optimizer = GetComponent<ESOptimizer>();
        optimizable = GetComponent<IESOptimizable>();
        Debug.Assert(optimizable != null, "DesicionMAES need to attach to a gameobjec with an agent that implements IESOptmizable.");

    }

    public override float[] Decide(List<float> vectorObs, List<Texture2D> visualObs, List<float> heuristicAction, List<float> heuristicVariance = null)
    {
        
       
        if (heuristicVariance != null && useHeuristic)
            optimizer.initialStepSize = heuristicVariance[0];
        double[] best = optimizer.Optimize(optimizable, useHeuristic?heuristicAction.Select(t=>(double)t).ToArray(): new double[heuristicAction.Count]);
        
        var result = Array.ConvertAll(best, t => (float)t);
        return result;
    }
    
}
