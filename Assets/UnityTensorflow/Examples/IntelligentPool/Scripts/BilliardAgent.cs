using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BilliardAgent : AgentES
{
    
    protected BilliardGameSystem gameSystem;
    public int shootSequence = 1;
    public bool resetAfterOneShot = false;
    public bool resetOnStart = false;
    protected Color visColor;

    public bool randomizeBalls = true;
    public bool autoRequestDecision = false;
    public bool AutoRequestDicision { get { return autoRequestDecision; } set{ autoRequestDecision = value; } }

    private bool hasShot = false;

    private void Start()
    {
        //test
        //gameSystem.ShootSequence(new List<Vector3>() { Vector3.right, Vector3.back });
        if(resetOnStart)
            gameSystem.Reset(randomizeBalls);
    }

    public override void InitializeAgent()
    {
        if(gameSystem == null)
            gameSystem = GetComponentInChildren<BilliardGameSystem>() as BilliardGameSystem;
        //gameSystem.Reset(randomizeRedballs);
    }

    public override void AgentReset()
    {
        
    }

    public override void AgentOnDone()
    {
    }

    public override void CollectObservations()
    {
        var balls = gameSystem.GetBallsStatus();
        foreach(var b in balls)
        {
            AddVectorObs(b);
        }
    }



    private void FixedUpdate()
    {
        bool doReset = false;
        if (gameSystem.GameComplete() || (gameSystem.AllShotsComplete() && hasShot && resetAfterOneShot))
        {
            gameSystem.Reset(randomizeBalls);
            doReset = true;
            hasShot = false;
        }

        if(autoRequestDecision && gameSystem.AllShotsComplete() && !doReset)
        {
            AddReward(gameSystem.defaultArena.ActualScore);
            gameSystem.bestScore = Mathf.NegativeInfinity;
            RequestDecision();
            hasShot = true;
        }
    }


    public override List<float> Evaluate(List<double[]> action)
    {

        List<List<Vector3>> forceSequences = new List<List<Vector3>>();
        for (int i = 0; i < action.Count; ++i)
        {
            int seq = action[i].Length / 2;
            forceSequences.Add(new List<Vector3>());
            for (int j = 0; j < seq; ++j)
            {
                double[] act = new double[2];
                Array.Copy(action[i], 2*j, act, 0, 2);

                forceSequences[i].Add(ParamsToForceVector(act));
            }
        }
        var values = gameSystem.EvaluateShotSequenceBatch(forceSequences, Color.gray);
        return values;
    }

    public override void OnReady(double[] vectorAction)
    {
        int seq = vectorAction.Length / 2;
        var result = new List<Vector3>();
        for (int j = 0; j < seq; ++j)
        {
            double[] act = new double[2];
            Array.Copy(vectorAction, 2 * j, act, 0, 2);

            result.Add(ParamsToForceVector(act));
        }

        //print("Shoot with params:" + string.Join(",",vectorAction));

        //gameSystem.Shoot(ParamsToForceVector(vectorAction));
        gameSystem.ShootSequence(result);
        Physics.autoSimulation = true;

        //test
        //AddReward(-Mathf.Abs((float)vectorAction[0]) + Mathf.Abs((float)vectorAction[1]));
    }




    public Vector3 ParamsToForceVector(double[] x)
    {
        Vector3 force = (new Vector3((float)x[0], 0, (float)x[1]));
        //if (force.magnitude > maxForce)
            //force = maxForce * force.normalized;
        return force;
    }
    public Vector3 SamplePointToForceVectorRA(float x, float y)
    {
        x = Mathf.Clamp01(x); y = Mathf.Clamp01(y);
        float angle = x * Mathf.PI*2;
        float force = y ;
        double[] param = new double[2];
        param[0] = Mathf.Sin(angle) * force ;
        param[1] = Mathf.Cos(angle) * force ;
        return ParamsToForceVector(param);
    }

    public Vector3 SamplePointToForceVectorXY(float x, float y)
    {
        x = Mathf.Clamp01(x); y = Mathf.Clamp01(y);
        float fx = x - 0.5f;
        float fy = y - 0.5f;

        double[] param = new double[2];
        param[0] = fx*2;
        param[1] = fy * 2;
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
