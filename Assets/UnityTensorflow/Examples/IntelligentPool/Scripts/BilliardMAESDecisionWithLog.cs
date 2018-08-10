using System.Collections;
using System.Collections.Generic;
using MLAgents;
using UnityEngine;

public class BilliardMAESDecisionWithLog : DecisionMAES
{

    public bool log = true;
    public int logInterval = 20;

    protected StatsLogger logger = new StatsLogger(); 

    public override float[] Decide(List<float> vectorObs, List<Texture2D> visualObs, List<float> heuristicAction, List<float> heuristicVariance = null)
    {
        var result =  base.Decide(vectorObs, visualObs, heuristicAction, heuristicVariance);
        logger.AddData("Average MAES iteration",optimizer.Iteration, logInterval);
        logger.AddData("Average MAES best Score", (float)optimizer.BestScore, logInterval);
        return result;
    }
}
