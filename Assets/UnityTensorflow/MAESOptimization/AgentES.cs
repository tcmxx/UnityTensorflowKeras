using ICM;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;
using System;
using System.Linq;

[RequireComponent(typeof(ESOptimizer))]
public abstract class AgentES : Agent, IESOptimizable
{


    public ESOptimizer Optimizer { get; protected set; }

    private void Awake()
    {
        Optimizer = GetComponent<ESOptimizer>();
    }



    //for asynchronized decision, set this to false.
    public bool synchronizedDecision = true;
    public event System.Action<AgentES> OnEndOptimizationRequested;
    /// <summary>
    /// return the value of an action.
    /// </summary>
    /// <param name="action"></param>
    /// <returns></returns>
    public abstract List<float> Evaluate(List<double[]> action);


    /// <summary>
    /// Implement this instead  of AgentAction()
    /// </summary>
    /// <param name="vectorAction"></param>
    public abstract void OnReady(double[] vectorAction);


    public enum VisualizationMode
    {
        Sampling,
        Best,
        None
    }
    public virtual void SetVisualizationMode(VisualizationMode visMode) { }

    /// <summary>
    /// Don't override this method, implement OnActionReady() instead.
    /// Set synchronizedDecision to true if you want this agent to react when calling AgentAction
    /// </summary>
    /// <param name="vectorAction"></param>
    /// <param name="textAction"></param>
    public override void AgentAction(float[] vectorAction, string textAction)
    {
        if (synchronizedDecision)
            OnReady(Array.ConvertAll(vectorAction, t => (double)t));
    }
    
    public double[] Optimize(double[] initialMean = null)
    {
        return Optimizer.Optimize(this, initialMean);
    }

    public void OptimizeAsync(double[] initialMean = null)
    {
        Optimizer.StartOptimizingAsync(this, OnReady,initialMean);
    }

    public void ForceEndOptimization()
    {
        Optimizer.StopOptimizing(OnReady);
    }

    public int GetParamDimension()
    {
        return brain.brainParameters.vectorActionSize[0];
    }
}
