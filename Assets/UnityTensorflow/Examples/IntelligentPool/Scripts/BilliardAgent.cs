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




    public override float Evaluate(double[] action)
    {
        return gameSystem.evaluateShot(ParamsToForceVector(action), visColor);
    }

    public override void OnReady(double[] vectorAction)
    {
        gameSystem.shoot(ParamsToForceVector(vectorAction));
    }




    public Vector3 ParamsToForceVector(double[] x)
    {
        Vector3 force = forceMultiplier * (new Vector3((float)x[0], 0, (float)x[1]));
        if (force.magnitude > maxForce)
            force = maxForce * force.normalized;
        return force;
    }
    public Vector3 SamplePointToForceVectorRA(float x, float y)
    {
        x = Mathf.Clamp01(x); y = Mathf.Clamp01(y);
        float angle = x * Mathf.PI*2;
        float force = y * maxForce;
        double[] param = new double[2];
        param[0] = Mathf.Sin(angle) * force / forceMultiplier;
        param[1] = Mathf.Cos(angle) * force / forceMultiplier;
        return ParamsToForceVector(param);
    }

    public Vector3 SamplePointToForceVectorXY(float x, float y)
    {
        x = Mathf.Clamp01(x); y = Mathf.Clamp01(y);
        float fx = x - 0.5f;
        float fy = y - 0.5f;

        double[] param = new double[2];
        param[0] = fx*2 * maxForce / forceMultiplier;
        param[1] = fy * 2 * maxForce / forceMultiplier;
        return ParamsToForceVector(param);
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
