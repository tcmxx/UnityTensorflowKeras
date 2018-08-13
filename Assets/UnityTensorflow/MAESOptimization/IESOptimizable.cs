using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public interface IESOptimizable {

    /// <summary>
    /// evaluate a batch of params
    /// </summary>
    /// <param name="param"></param>
    /// <returns></returns>
    List<float> Evaluate(List<double[]> param);

    /// <summary>
    /// Return the dimension of the parameters
    /// </summary>
    /// <returns>dimension of the parameters</returns>
    int GetParamDimension();
}
