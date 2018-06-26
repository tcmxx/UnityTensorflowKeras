using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public abstract class AgentDependentDecision : MonoBehaviour
{
   public  abstract float[] Decide(Agent agent, List<float> vectorObs, List<Texture2D> visualObs, List<float> heuristicAction);
}