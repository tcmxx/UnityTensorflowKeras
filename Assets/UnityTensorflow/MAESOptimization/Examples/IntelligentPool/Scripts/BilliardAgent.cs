using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BilliardAgent : AgentES
{
    
    protected BilliardGameSystem gameSystem;
    public float forceMultiplier = 100;
    public float maxForce = 5;
    protected Color visColor;

    public override void InitializeAgent()
    {
        gameSystem = FindObjectOfType(typeof(BilliardGameSystem)) as BilliardGameSystem;
    }

    public override void AgentReset()
    {
    }

    public override void AgentOnDone()
    {

    }




    public override float EvaluateAction(double[] action)
    {
        return gameSystem.evaluateShot(ParamsToForceVector(action), visColor);
    }

    public override void OnActionReady(double[] vectorAction)
    {
        gameSystem.shoot(ParamsToForceVector(vectorAction));
    }




    protected Vector3 ParamsToForceVector(double[] x)
    {
        Vector3 force = forceMultiplier * (new Vector3((float)x[0], 0, (float)x[1]));
        if (force.magnitude > maxForce)
            force = maxForce * force.normalized;
        return force;
    }

    public override void SetVisualizationMode(VisualizationMode visMode)
    {
        if(visMode == VisualizationMode.Best)
        {
            visColor = Color.green;
        }else if(visMode == VisualizationMode.Sampling)
        {
            visColor = Color.grey;
        }
        else
        {
            visColor = new Color(0, 0, 0, 0);
        }
    }
}
