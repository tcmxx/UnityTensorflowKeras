﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using ICM;

#if UNITY_EDITOR
using UnityEditor;
#endif


[CreateAssetMenu(menuName = "ML-Agents/InternalLearning/neural evolution/TrainerParamsNeuralEvolution")]
public class TrainerParamsNeuralEvolution : TrainerParams
{
    [Header("Optimization")]
    public ESOptimizer.ESOptimizerType optimizerType;
    
    public int sampleCountForEachChild = 10;
    public int populationSize = 16;
    public OptimizationModes mode;
    public float initialStepSize = 1;
    public int timeHorizon = 500;
    
}
