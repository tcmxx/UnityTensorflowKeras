using System.Collections;
using System.Collections.Generic;
using MLAgents;
using UnityEngine;

public class BilliardMAESDecisionWithLog : DecisionMAES
{

    public bool log = true;
    public int logInterval = 20;

    protected StatsLogger logger = new StatsLogger();
    protected int logStep = 0;
    public override float[] Decide(List<float> vectorObs, List<float[,,]> visualObs, List<float> heuristicAction, List<float> heuristicVariance = null)
    {
        var result =  base.Decide(vectorObs, visualObs, heuristicAction, heuristicVariance);
        if (log)
        {
            logStep++;
            logger.AddData("Average MAES iteration", optimizer.Iteration);
            logger.AddData("Average MAES best Score", (float)optimizer.BestScore);
            if(logStep%logInterval == 0)
            {
                logger.LogAllCurrentData(logStep);
            }
        }
        return result;
    }
}
