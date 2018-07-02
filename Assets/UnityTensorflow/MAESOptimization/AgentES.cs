using ICM;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;
using System;
using System.Linq;

public abstract class AgentES : Agent, IESOptimizable
{
    //for asynchronized decision, set this to false.
    public bool callOnReadyInAgentAction = true;
    public int maxIteration;
    public float targetValue;
    public int populationSize = 16;
    public float initialStepSize = 1;

    public event System.Action<AgentES> OnEndOptimizationRequested;
    /// <summary>
    /// return the value of an action.
    /// </summary>
    /// <param name="action"></param>
    /// <returns></returns>
    public abstract List<float> Evaluate(List<double[]> action);


    /// <summary>
    /// Implement this instead 
    /// </summary>
    /// <param name="vectorAction"></param>
    public abstract void OnReady(double[] vectorAction);


    public enum VisualizationMode
    {
        Sampling,
        Best,
        None
    }
    public abstract void SetVisualizationMode(VisualizationMode visMode);

    /// <summary>
    /// Don't override this method, implement OnActionReady() instead.
    /// </summary>
    /// <param name="vectorAction"></param>
    /// <param name="textAction"></param>
    public override void AgentAction(float[] vectorAction, string textAction)
    {
        if (callOnReadyInAgentAction)
            OnReady(Array.ConvertAll(vectorAction, t => (double)t));
    }
    

    public void ForceEndOptimization()
    {
        OnEndOptimizationRequested.Invoke(this);
    }

    public int GetParamDimension()
    {
        return brain.brainParameters.vectorActionSize;
    }
}
