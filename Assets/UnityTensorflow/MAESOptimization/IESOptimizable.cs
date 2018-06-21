using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public interface IESOptimizable {

    /// <summary>
    /// return the value of an action.
    /// </summary>
    /// <param name="param"></param>
    /// <returns></returns>
    float Evaluate(double[] param);


    /// <summary>
    /// Implement this instead 
    /// </summary>
    /// <param name="param"></param>
    void OnReady(double[] param);

    int GetParamDimension();
}
