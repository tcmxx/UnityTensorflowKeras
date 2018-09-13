using Accord.Math;
using Accord.Statistics.Distributions.Univariate;
using KerasSharp;
using KerasSharp.Backends;
using KerasSharp.Engine.Topology;
using KerasSharp.Initializers;
using KerasSharp.Losses;
using KerasSharp.Models;
using MLAgents;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using static KerasSharp.Backends.Current;

#if UNITY_EDITOR
using UnityEditor;
#endif









public class RLModelPPOHierarchy : RLModelPPO {

    [ShowAllPropertyAttr()]
    public RLNetowrkACHierarchy networkHierarchy;
    
    public int lowLevelObservationSize;
    public int highLevelObservationSize;

    //some holders for tensors
    protected Tensor inputLowLevelTensor = null;
    protected Tensor inputHighLevelTensor = null;

    /// <summary>
    /// Initialize the model without training parts
    /// </summary>
    /// <param name="brainParameters"></param>
    public override void InitializeInner(BrainParameters brainParameters, Tensor stateTensor, List<Tensor> visualTensors, TrainerParams trainerParams)
    {

        Debug.Assert(visualTensors == null, "RLModelPPOHierarchy does not support visual input yet");


        if (highLevelObservationSize > 0)
        {
            var splited = K.split(stateTensor, K.constant(new int[] { lowLevelObservationSize, highLevelObservationSize }, dtype: DataType.Int32), K.constant(1, dtype: DataType.Int32), 2);
            inputLowLevelTensor = splited[0];
            inputHighLevelTensor = splited[1];
        }
        else
        {
            inputLowLevelTensor = stateTensor;
        }
        
        List<Tensor> inputVisualTensors = visualTensors;

        if (useInputNormalization && HasVectorObservation)
        {
            inputLowLevelTensor = CreateRunninngNormalizer(inputLowLevelTensor, StateSize);
        }



        Tensor outputValue = null; Tensor outputAction = null; Tensor outputVariance = null;
        Debug.Assert(ActionSize.Length <= 1, "Action branching is not supported yet");
        //build the network
        networkHierarchy.BuildNetwork(inputLowLevelTensor, inputHighLevelTensor, ActionSize[0], ActionSpace, out outputAction, out outputValue, out outputVariance);

        InitializePPOStructures(trainerParams, stateTensor, inputVisualTensors, outputValue, outputAction, outputVariance, networkHierarchy.GetHighLevelWeights());

    }
    public override List<Tensor> GetAllModelWeights()
    {
        var result = new List<Tensor>();
        result.AddRange(networkHierarchy.GetHighLevelWeights()); result.AddRange(networkHierarchy.GetLowLevelWeights());
        return result;
    }


}
