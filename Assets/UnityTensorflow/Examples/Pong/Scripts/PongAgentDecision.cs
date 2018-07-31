using System.Collections;
using System.Collections.Generic;
using MLAgents;
using UnityEngine;

public class PongAgentDecision : AgentDependentDecision {
    public override float[] Decide(Agent agent, List<float> vectorObs, List<Texture2D> visualObs, List<float> heuristicAction)
    {
        if (agent.brain.brainParameters.vectorActionSpaceType == SpaceType.discrete)
        {
            float[] result = new float[1];
            result[0] = vectorObs[0] > vectorObs[3] ? 0 : 2;
            return result;
        }
        else
        {
            float[] result = new float[1];
            result[0] = vectorObs[0] > vectorObs[3] ? -1 : 1;
            return result;
        }
    }
}
