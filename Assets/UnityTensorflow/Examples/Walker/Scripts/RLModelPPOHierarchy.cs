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

        Tensor inputStateTensor = stateTensor;
        List<Tensor> inputVisualTensors = visualTensors;

        if (useInputNormalization && HasVectorObservation)
        {
            inputStateTensor = CreateRunninngNormalizer(inputStateTensor, StateSize);
        }



        Tensor outputValue = null; Tensor outputAction = null; Tensor outputVariance = null;
        //build the network
        networkHierarchy.BuildNetwork(inputLowLevelTensor, inputHighLevelTensor, ActionSize, ActionSpace, out outputAction, out outputValue, out outputVariance);

        //actor network output variance
        /*if (ActionSpace == SpaceType.continuous)
        {
            logSigmaSq = K.variable((new Constant(0)).Call(new int[] { ActionSize }, DataType.Float), name: "PPO.log_sigma_square");
            outputVariance = K.exp(logSigmaSq);
        }*/

        List<Tensor> allobservationInputs = new List<Tensor>();
        if (HasVectorObservation)
        {
            allobservationInputs.Add(inputStateTensor);
        }
        if (HasVisualObservation)
        {
            allobservationInputs.AddRange(inputVisualTensors);
        }

        ValueFunction = K.function(allobservationInputs, new List<Tensor> { outputValue }, null, "ValueFunction");
        if (ActionSpace == SpaceType.continuous)
        {
            ActionFunction = K.function(allobservationInputs, new List<Tensor> { outputAction, outputVariance }, null, "ActionFunction");
        }
        else
        {
            ActionFunction = K.function(allobservationInputs, new List<Tensor> { outputAction }, null, "ActionFunction");
        }

        TrainerParamsPPO trainingParams = trainerParams as TrainerParamsPPO;
        if (trainingParams != null)
        {
            //training needed inputs
            var inputAction = UnityTFUtils.Input(new int?[] { ActionSpace == SpaceType.continuous ? ActionSize : 1 }, name: "InputAction", dtype: ActionSpace == SpaceType.continuous ? DataType.Float : DataType.Int32)[0];
            var inputOldProb = UnityTFUtils.Input(new int?[] { ActionSpace == SpaceType.continuous ? ActionSize : 1 }, name: "InputOldProb")[0];
            var inputAdvantage = UnityTFUtils.Input(new int?[] { 1 }, name: "InputAdvantage")[0];
            var inputTargetValue = UnityTFUtils.Input(new int?[] { 1 }, name: "InputTargetValue")[0];
            var inputOldValue = UnityTFUtils.Input(new int?[] { 1 }, name: "InputOldValue")[0];

            ClipEpsilon = trainingParams.clipEpsilon;
            ValueLossWeight = trainingParams.valueLossWeight;
            EntropyLossWeight = trainingParams.entropyLossWeight;

            var inputClipEpsilon = UnityTFUtils.Input(batch_shape: new int?[] { }, name: "ClipEpsilon", dtype: DataType.Float)[0];
            var inputValuelossWeight = UnityTFUtils.Input(batch_shape: new int?[] { }, name: "ValueLossWeight", dtype: DataType.Float)[0];
            var inputEntropyLossWeight = UnityTFUtils.Input(batch_shape: new int?[] { }, name: "EntropyLossWeight", dtype: DataType.Float)[0];

            // action probability from input action
            Tensor outputEntropy;
            Tensor actionProb;
            using (K.name_scope("ActionProb"))
            {
                if (ActionSpace == SpaceType.continuous)
                {
                    var temp = K.mul(outputVariance, 2 * Mathf.PI * 2.7182818285);
                    temp = K.mul(K.log(temp), 0.5);
                    if (outputVariance.shape.Length == 2)
                    {
                        outputEntropy = K.mean(K.mean(temp, 0, false), name: "OutputEntropy");
                    }
                    else
                    {
                        outputEntropy = K.mean(temp, 0, false, name: "OutputEntropy");
                    }

                    actionProb = K.normal_probability(inputAction, outputAction, outputVariance);
                }
                else
                {
                    var onehotInputAction = K.one_hot(inputAction, K.constant<int>(ActionSize, dtype: DataType.Int32), K.constant(1.0f), K.constant(0.0f));
                    onehotInputAction = K.reshape(onehotInputAction, new int[] { -1, ActionSize });
                    outputEntropy = K.mean((-1.0f) * K.sum(outputAction * K.log(outputAction + 0.00000001f), axis: 1), 0);
                    actionProb = K.reshape(K.sum(outputAction * onehotInputAction, 1), new int[] { -1, 1 });
                }
            }

            // value loss
            var clippedValueEstimate = inputOldValue + K.clip(outputValue - inputOldValue, 0.0f - inputClipEpsilon, inputClipEpsilon);
            var valueLoss1 = new MeanSquareError().Call(outputValue, inputTargetValue);
            var valueLoss2 = new MeanSquareError().Call(clippedValueEstimate, inputTargetValue);
            var outputValueLoss = K.mean(K.maximum(valueLoss1, valueLoss2));


            // Clipped Surrogate loss
            Tensor outputPolicyLoss;
            using (K.name_scope("ClippedCurreogateLoss"))
            {
                var probRatio = actionProb / (inputOldProb + 0.0000001f);
                var p_opt_a = probRatio * inputAdvantage;
                var p_opt_b = K.clip(probRatio, 1.0f - inputClipEpsilon, 1.0f + inputClipEpsilon) * inputAdvantage;

                outputPolicyLoss = K.mean(1 - K.mean(K.minimun(p_opt_a, p_opt_b)), name: "ClippedCurreogateLoss");
            }
            //final weighted loss
            var outputLoss = outputPolicyLoss + inputValuelossWeight * outputValueLoss;
            outputLoss = outputLoss - inputEntropyLossWeight * outputEntropy;


            //add inputs, outputs and parameters to the list
            List<Tensor> updateParameters = network.GetWeights();
            List<Tensor> allInputs = new List<Tensor>();
            if (HasVectorObservation)
            {
                allInputs.Add(inputStateTensor);
            }
            if (HasVisualObservation)
            {
                allInputs.AddRange(inputVisualTensors);
            }
            allInputs.Add(inputAction);
            allInputs.Add(inputOldProb);
            allInputs.Add(inputTargetValue);
            allInputs.Add(inputOldValue);
            allInputs.Add(inputAdvantage);
            allInputs.Add(inputClipEpsilon);
            allInputs.Add(inputValuelossWeight);
            allInputs.Add(inputEntropyLossWeight);

            //create optimizer and create necessary functions
            var updates = AddOptimizer(updateParameters, outputLoss, optimizer);
            UpdateFunction = K.function(allInputs, new List<Tensor> { outputLoss, outputValueLoss, outputPolicyLoss, outputEntropy }, updates, "UpdateFunction");
        }


    }
    public override List<Tensor> GetAllModelWeights()
    {
        var result = new List<Tensor>();
        result.AddRange(networkHierarchy.GetHighLevelWeights()); result.AddRange(networkHierarchy.GetLowLevelWeights());
        return result;
    }


}
