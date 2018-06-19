using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public interface IESOptimizable {

    /// <summary>
    /// return the value of an action.
    /// </summary>
    /// <param name="action"></param>
    /// <returns></returns>
    float EvaluateAction(double[] action);


    /// <summary>
    /// Implement this instead 
    /// </summary>
    /// <param name="vectorAction"></param>
    void OnActionReady(double[] vectorAction);

   
}
