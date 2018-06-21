using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BilliardSimple : MonoBehaviour, IESOptimizable
{
    
    protected BilliardGameSystem gameSystem;
    public float forceMultiplier = 100;
    public float maxForce = 5;
    public ESOptimizer optimizer;
    
    private void Start()
    {
        gameSystem = FindObjectOfType(typeof(BilliardGameSystem)) as BilliardGameSystem;
        Debug.Assert(gameSystem != null, "Did not find BilliardGameSystem in the scene");
    }

    private void Update()
    {
        if (optimizer.IsOptimizing)
        {
            gameSystem.evaluateShot(ParamsToForceVector(optimizer.BestParams), Color.green);
        }
    }

    public float Evaluate(double[] action)
    {
        return gameSystem.evaluateShot(ParamsToForceVector(action), Color.grey);
    }

    public void OnReady(double[] vectorAction)
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

    public int GetParamDimension()
    {
        return 2;
    }
}
