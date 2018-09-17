using KerasSharp.Engine.Topology;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public interface INeuralEvolutionModel
{
    float[,] EvaluateAction(float[,] vectorObservation, List<float[,,,]> visualObservation, List<float[,]> actionsMask = null);
    List<Tensor> GetWeightsForNeuralEvolution();
}