using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class PongAgent : Agent
{
    [HideInInspector]
    public PongEnvironment environment;

    public override void InitializeAgent()
    {
    }

    public override void CollectObservations()
    {
        AddVectorObs(environment.CurrentState(this));
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        environment.MoveRacket(this, vectorAction[0]);

        //test if the learning is correct
        /*var states = environment.CurrentState(this);
        if (states[0] > states[3] && Mathf.RoundToInt(vectorAction[0]) == 0)
        {
            //AddReward(1);
        }
        else if(states[0] < states[3] && Mathf.RoundToInt(vectorAction[0]) == 2)
        {
           // AddReward(1);
        }*/
    }

    public override void AgentReset()
    {
        environment.Reset();
    }
}
