using KerasSharp.Engine.Topology;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

public abstract class GANNetwork : UnityNetwork
{
    /// <summary>
    /// 
    /// </summary>
    /// <param name="inputCondition"></param>
    /// <param name="inputNoise"></param>
    /// <param name="inputTargetToJudge"></param>
    /// <param name="outputShape">Output shape without batch dimension</param>
    /// <param name="generatorOutput"></param>
    /// <param name="generatorLoss"></param>
    /// <param name="discriminatorLoss"></param>
    public abstract void BuildNetwork(Tensor inputCondition, Tensor inputNoise, Tensor inputTargetToJudge, int[] outputShape, out Tensor generatorOutput, out Tensor discriminatorOutputExternal, out Tensor discriminatorOutputFromGenerators);
    public abstract List<Tensor> GetGeneratorWeights();
    public abstract List<Tensor> GetDiscriminatorWeights();
}
