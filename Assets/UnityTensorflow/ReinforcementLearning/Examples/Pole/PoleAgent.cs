using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PoleAgent : Agent
{
    public float deltaTime = 0.02f;
    public float damp = 0.01f;
    public float gravity = 1;


    public float velR;
    public float angleR;    //in radian

    public GameObject poleObjectRef;
    public bool noVectorObservation = false;
    private void Update()
    {
        poleObjectRef.transform.rotation = Quaternion.Euler(0, 0, angleR * Mathf.Rad2Deg + 180);
    }


    public override void InitializeAgent()
    {

    }

    public override void CollectObservations()
    {
        if (!noVectorObservation)
        {
            AddVectorObs(velR);
            AddVectorObs(angleR);
        }
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        
        float gavityTorque = Mathf.Sin(angleR);
        float torque = 0;
        float dampeTorque = -velR * velR * damp * Mathf.Sign(velR);
        if (brain.brainParameters.vectorActionSpaceType == SpaceType.continuous)
        {
            torque = vectorAction[0];
            torque = Mathf.Clamp(torque, -3, 3);
        }
        else
        {
            torque = (vectorAction[0] == 0 ? -3.0f : 3.0f);
        }
        velR += deltaTime * (torque + gavityTorque + dampeTorque);
        angleR += deltaTime * velR;

        if (angleR < -Mathf.PI)
        {
            angleR = angleR + 2 * Mathf.PI;
        }
        else if (angleR > Mathf.PI)
        {
            angleR = angleR - 2 * Mathf.PI;
        }

        float reward = -Mathf.Abs(angleR) + Mathf.PI / 2 - Mathf.Abs(velR);
        reward /= 10;

        SetReward(reward);
    }

    public override void AgentReset()
    {
        angleR = Random.Range(-Mathf.PI / 2, Mathf.PI / 2);
        velR = Random.Range(0, 0);

    }

}
