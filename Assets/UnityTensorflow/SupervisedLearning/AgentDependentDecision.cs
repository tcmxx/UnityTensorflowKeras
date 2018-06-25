using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public abstract class AgentDependentDecision : MonoBehaviour
{
   public  abstract float[] Decide(List<float> vectorObs, List<Texture2D> visualObs, float reward, bool done, Agent agent);
}