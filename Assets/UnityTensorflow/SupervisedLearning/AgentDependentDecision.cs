using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public abstract class AgentDependentDecision : MonoBehaviour
{
    /// <summary>
    /// 
    /// </summary>
    /// <param name="agent"></param>
    /// <param name="vectorObs"></param>
    /// <param name="visualObs"></param>
    /// <param name="heuristicAction"></param>
    /// <param name="isTraining"></param>
    /// <returns></returns>
   public  abstract float[] Decide(Agent agent, List<float> vectorObs, List<Texture2D> visualObs, List<float> heuristicAction);
}