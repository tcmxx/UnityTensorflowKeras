using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Accord.Statistics.Distributions.Univariate;
using System;
using System.Linq;
using Accord;
using Accord.Math;
using Accord.Statistics;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;
#if UNITY_EDITOR
using UnityEditor;
#endif


using static KerasSharp.Backends.Current;
using KerasSharp.Backends;

using MLAgents;
using KerasSharp.Models;
using KerasSharp.Optimizers;
using KerasSharp.Engine.Topology;
using KerasSharp.Initializers;
using KerasSharp;
using KerasSharp.Losses;


public interface IRLModelPPOCMA
{
    float ClipEpsilonValue { get; set; }

    float[] EvaluateValue(float[,] vectorObservation, List<float[,,,]> visualObservation);
    float[,] EvaluateAction(float[,] vectorObservation, List<float[,,,]> visualObservation);

    float[] TrainValue(float[,] vectorObservations, List<float[,,,]> visualObservations, float[] oldValues, float[] targetValues);
    float[] TrainMean(float[,] vectorObservations, List<float[,,,]> visualObservations, float[,] actions, float[] advantages);
    float[] TrainVariance(float[,] vectorObservations, List<float[,,,]> visualObservations, float[,] actions, float[] advantages);
    float[] Pretrain(float desiredStd, float desiredMean, float[] inputStds, float[] inputMeans, int batchSize = 64);
}


public class RLModelPPOCMA : LearningModelBase, IRLModelPPOCMA
{


    protected Function ValueFunction { get; set; }
    protected Function ActionFunction { get; set; }
    protected Function TrainMeanFunction { get; set; }
    protected Function TrainVarianceFunction { get; set; }
    protected Function TrainValueFunction { get; set; }

    protected Function PretrainFunction { get; set; }
    protected Function UpdateNormalizerFunction { get; set; }

    [ShowAllPropertyAttr]
    public RLNetworkACSeperateVar network;

    public OptimizerCreator optimizerValue;
    public OptimizerCreator optimizerMean;
    public OptimizerCreator optimizerVariance;

    public OptimizerCreator optimizerPretrain;

    public bool useInputNormalization = false;
    public bool usePositiveAdvOnly = true;
    public float ClipEpsilonValue { get; set; }

    //the variables for normalization
    protected Tensor runningMean = null;
    protected Tensor runningVariance = null;
    protected Tensor stepCount = null;

    /// <summary>
    /// Initialize the model without training parts
    /// </summary>
    /// <param name="brainParameters"></param>
    public override void InitializeInner(BrainParameters brainParameters, Tensor stateTensor, List<Tensor> visualTensors, TrainerParams trainerParams)
    {
        Debug.Assert(ActionSpace == SpaceType.continuous, "RLModelPPOCMA only support continuous action space.");
        Tensor inputStateTensorToNetwork = stateTensor;

        if (useInputNormalization && HasVectorObservation)
        {
            inputStateTensorToNetwork = CreateRunninngNormalizer(inputStateTensorToNetwork, StateSize);
        }


        //build the network
        Tensor outputValue = null; Tensor outputAction = null; Tensor outActionLogVariance = null;
        network.BuildNetworkForContinuousActionSapce(inputStateTensorToNetwork, visualTensors, null, null, ActionSizes[0], out outputAction, out outputValue, out outActionLogVariance);


        InitializePPOCMAStructures(trainerParams, stateTensor, visualTensors, outputValue, outputAction, outActionLogVariance, network.GetCriticWeights(), network.GetActorMeanWeights(), network.GetActorVarianceWeights());

    }




    /// <summary>
    /// Initialize the model for PPO
    /// </summary>
    /// <param name="trainerParams"></param>
    /// <param name="stateTensor"></param>
    /// <param name="inputVisualTensors"></param>
    /// <param name="outputValueFromNetwork"></param>
    /// <param name="outputActionFromNetwork"></param>
    /// <param name="outputVarianceFromNetwork"></param>
    protected void InitializePPOCMAStructures(TrainerParams trainerParams, Tensor stateTensor, List<Tensor> inputVisualTensors, Tensor outputValueFromNetwork, Tensor outputActionMeanFromNetwork, Tensor outActionLogVarianceFromNetwork, List<Tensor> valueWeights, List<Tensor> meanWeights, List<Tensor> varweights)
    {
        List<Tensor> allobservationInputs = new List<Tensor>();
        if (HasVectorObservation)
        {
            allobservationInputs.Add(stateTensor);
        }
        if (HasVisualObservation)
        {
            allobservationInputs.AddRange(inputVisualTensors);
        }

        ValueFunction = K.function(allobservationInputs, new List<Tensor> { outputValueFromNetwork }, null, "ValueFunction");

        Tensor outputActualAction = null;
        Tensor outputVariance = K.exp(outActionLogVarianceFromNetwork);
        using (K.name_scope("SampleAction"))
        {
            outputActualAction = K.standard_normal(K.shape(outputActionMeanFromNetwork), DataType.Float) * K.sqrt(outputVariance) + outputActionMeanFromNetwork;

        }

        ActionFunction = K.function(allobservationInputs, new List<Tensor> { outputActualAction, outputActionMeanFromNetwork, outputVariance }, null, "ActionFunction");

        TrainerParamsPPO trainingParams = trainerParams as TrainerParamsPPO;
        if (trainingParams != null)
        {
            //training needed inputs
            var inputOldAction = UnityTFUtils.Input(new int?[] { ActionSizes[0] }, name: "InputOldAction")[0];
            var inputAdvantage = UnityTFUtils.Input(new int?[] { 1 }, name: "InputAdvantage")[0];
            var inputTargetValue = UnityTFUtils.Input(new int?[] { 1 }, name: "InputTargetValue")[0];
            var inputOldValue = UnityTFUtils.Input(new int?[] { 1 }, name: "InputOldValue")[0];

            //var inputClipEpsilon = UnityTFUtils.Input(batch_shape: new int?[] { }, name: "ClipEpsilon", dtype: DataType.Float)[0];

            var inputClipEpsilonValue = UnityTFUtils.Input(batch_shape: new int?[] { }, name: "ClipEpsilonValue", dtype: DataType.Float)[0];
            // value loss   
            Tensor outputValueLoss = null;
            using (K.name_scope("ValueLoss"))
            {
                var clippedValueEstimate = inputOldValue + K.clip(outputValueFromNetwork - inputOldValue, 0.0f - inputClipEpsilonValue, inputClipEpsilonValue);
                var valueLoss1 = new MeanSquareError().Call(outputValueFromNetwork, inputTargetValue);
                var valueLoss2 = new MeanSquareError().Call(clippedValueEstimate, inputTargetValue);
                outputValueLoss = K.mean(K.maximum(valueLoss1, valueLoss2));
                outputValueLoss = K.mean(valueLoss1);
            }

            var valueUpdates = AddOptimizer(valueWeights, outputValueLoss, optimizerValue);
            List<Tensor> valueInputs = new List<Tensor>();
            if (HasVectorObservation)
            {
                valueInputs.Add(stateTensor);
            }
            if (HasVisualObservation)
            {
                valueInputs.AddRange(inputVisualTensors);
            }
            valueInputs.Add(inputOldValue);
            valueInputs.Add(inputTargetValue);
            valueInputs.Add(inputClipEpsilonValue);
            TrainValueFunction = K.function(valueInputs, new List<Tensor> { outputValueLoss }, valueUpdates, "TrainValueFunction");

            // actor losses
            Tensor meanLoss, varLoss;
            using (K.name_scope("ActorLosses"))
            {
                Tensor posAdvantage;
                if (usePositiveAdvOnly)
                    posAdvantage = K.identity(K.relu(K.mean(inputAdvantage)), "ClipedPositiveAdv");
                else
                    posAdvantage = K.identity(K.mean(inputAdvantage), "Adv");
                var meanNoGrad = K.stop_gradient(outputActionMeanFromNetwork, "MeanNoGrad");
                var varNoGrad = K.stop_gradient(outputVariance, "VarNoGrad");
                var logVar = outActionLogVarianceFromNetwork;
                var logVarNoGrad = K.stop_gradient(logVar, "LogVarNoGrad");
                using (K.name_scope("VarLoss"))
                {
                    var logpNoMeanGrad = -1.0f*K.sum(0.5f * K.square(inputOldAction - meanNoGrad) / outputVariance + 0.5f * logVar, 1);
                    varLoss = K.identity(-1.0f * K.mean(posAdvantage * logpNoMeanGrad), "VarLoss");
                }
                using (K.name_scope("MeanLoss"))
                {
                    var logpNoVarGrad = -1.0f * K.sum(0.5f * K.square(inputOldAction - outputActionMeanFromNetwork) / varNoGrad + 0.5f * logVarNoGrad, 1);
                    meanLoss = K.identity(-1.0f * K.mean(posAdvantage * logpNoVarGrad), "MeanLoss");
                }


            }

            //add inputs, outputs and parameters to the list
            List<Tensor> allInputs = new List<Tensor>();
            if (HasVectorObservation)
            {
                allInputs.Add(stateTensor);
            }
            if (HasVisualObservation)
            {
                allInputs.AddRange(inputVisualTensors);
            }
            allInputs.Add(inputOldAction);
            allInputs.Add(inputAdvantage);


            //create optimizer and create necessary functions
            var updatesMean = AddOptimizer(meanWeights, meanLoss, optimizerMean);
            var updatesVar = AddOptimizer(varweights, varLoss, optimizerVariance);

            TrainMeanFunction = K.function(allInputs, new List<Tensor> { meanLoss }, updatesMean, "UpdateMeanFunction");
            TrainVarianceFunction = K.function(allInputs, new List<Tensor> { varLoss }, updatesVar, "UpdateMeanFunction");

            //pretraining for output mean and var
            var inputInitialStd = UnityTFUtils.Input(new int?[] { ActionSizes[0] }, name: "InputInitialStd")[0];
            var inputInitialMean = UnityTFUtils.Input(new int?[] { ActionSizes[0] }, name: "InputInitialMean")[0];
            var policyInitLoss = K.mean(K.mean(K.square(inputInitialMean - outputActionMeanFromNetwork)));
            policyInitLoss += K.mean(K.mean(K.square(inputInitialStd - K.sqrt(outputVariance))));
            
            var updatesPretrain = AddOptimizer(network.GetActorWeights(), policyInitLoss, optimizerPretrain);
            var pretrainInputs = new List<Tensor>();
            pretrainInputs.Add(stateTensor);
            pretrainInputs.Add(inputInitialMean);
            pretrainInputs.Add(inputInitialStd);
            PretrainFunction = K.function(pretrainInputs, new List<Tensor> { policyInitLoss }, updatesPretrain, "PretrainFunction");

        }
    }


    /// <summary>
    /// evaluate the value of current states
    /// </summary>
    /// <param name="vectorObservation">current vector observation. The first dimension of the array is the batch dimension.</param>
    /// <param name="visualObservation">current visual observation. The first dimension of the array is the batch dimension.</param>
    /// <returns>values of current states</returns>
    public virtual float[] EvaluateValue(float[,] vectorObservation, List<float[,,,]> visualObservation)
    {
        List<Array> inputLists = new List<Array>();
        if (HasVectorObservation)
        {
            Debug.Assert(vectorObservation != null, "Must Have vector observation inputs!");
            inputLists.Add(vectorObservation);
        }
        if (HasVisualObservation)
        {
            Debug.Assert(visualObservation != null, "Must Have visual observation inputs!");
            inputLists.AddRange(visualObservation);
        }

        var result = ValueFunction.Call(inputLists);
        //return new float[] { ((float[,])result[0].eval())[0,0] };
        var value = ((float[,])result[0].eval()).Flatten();
        return value;
    }

    /// <summary>
    /// Query actions based on curren states. The first dimension of the array must be batch dimension
    /// </summary>
    /// <param name="vectorObservation">current vector states. Can be batch input</param>
    /// <returns></returns>
    public virtual float[,] EvaluateAction(float[,] vectorObservation, List<float[,,,]> visualObservation)
    {
        List<Array> inputLists = new List<Array>();
        if (HasVectorObservation)
        {
            Debug.Assert(vectorObservation != null, "Must Have vector observation inputs!");
            inputLists.Add(vectorObservation);
        }
        if (HasVisualObservation)
        {
            Debug.Assert(visualObservation != null, "Must Have visual observation inputs!");
            inputLists.AddRange(visualObservation);
        }

        var result = ActionFunction.Call(inputLists);

        var outputAction = ((float[,])result[0].eval());
        float[,] actions = new float[outputAction.GetLength(0), outputAction.GetLength(1)];

        actions = outputAction;


        if (useInputNormalization && HasVectorObservation)
        {
            UpdateNormalizerFunction.Call(new List<Array>() { vectorObservation });
        }

        /*for(int i = 0; i < actions.GetLength(0); ++i)
        {
            for (int j = 0; j < actions.GetLength(1); ++j)
            {
                if (float.IsNaN(actions[i, j]))
                {
                    Debug.LogError("error");
                }
            }
        }*/

        return actions;

    }


    public float[] TrainValue(float[,] vectorObservations, List<float[,,,]> visualObservations, float[] oldValues, float[] targetValues)
    {
        Debug.Assert(TrainingEnabled == true, "The model needs to initalized with Training enabled to use TrainValue()");

        List<Array> inputs = new List<Array>();
        if (vectorObservations != null)
            inputs.Add(vectorObservations);
        if (visualObservations != null)
            inputs.AddRange(visualObservations);

        inputs.Add(oldValues);
        inputs.Add(targetValues);
        inputs.Add(new float[] { ClipEpsilonValue });

        var loss = TrainValueFunction.Call(inputs);
        var result = new float[] { (float)loss[0].eval() };
        return result;
    }


    public float[] TrainMean(float[,] vectorObservations, List<float[,,,]> visualObservations, float[,] actions, float[] advantages)
    {
        Debug.Assert(TrainingEnabled == true, "The model needs to initalized with Training enabled to use TrainValue()");

        List<Array> inputs = new List<Array>();
        if (vectorObservations != null)
            inputs.Add(vectorObservations);
        if (visualObservations != null)
            inputs.AddRange(visualObservations);

        inputs.Add(actions);
        inputs.Add(advantages);

        var loss = TrainMeanFunction.Call(inputs);
        var result = new float[] { (float)loss[0].eval() };
        return result;
    }
    public float[] TrainVariance(float[,] vectorObservations, List<float[,,,]> visualObservations, float[,] actions, float[] advantages)
    {
        Debug.Assert(TrainingEnabled == true, "The model needs to initalized with Training enabled to use TrainValue()");

        List<Array> inputs = new List<Array>();
        if (vectorObservations != null)
            inputs.Add(vectorObservations);
        if (visualObservations != null)
            inputs.AddRange(visualObservations);

        inputs.Add(actions);
        inputs.Add(advantages);

        var loss = TrainVarianceFunction.Call(inputs);
        var result = new float[] { (float)loss[0].eval() };
        return result;
    }

    public float[] Pretrain(float desiredStd, float desiredMean, float[] inputStds, float[] inputMeans, int batchSize = 64)
    {
        Debug.Assert(HasVectorObservation && !HasVisualObservation, "Pretrain only support vector observation");
        List<Array> inputList = new List<Array>();

        var obsDistributions = new NormalDistribution[StateSize];
        for(int i = 0; i < StateSize; ++i)
        {
            obsDistributions[i] = new NormalDistribution(inputMeans[i], inputStds[i] + 0.0000000001f);
        }

        float[,] vectorObservation = new float[batchSize, StateSize];
        for (int i = 0; i < batchSize; ++i)
        {
            for (int j = 0; j < StateSize; ++j)
            {
                vectorObservation[i, j] = (float)obsDistributions[j].Generate();
            }
        }

        inputList.Add(vectorObservation);
        float[,] means = new float[batchSize,ActionSizes[0]];
        float[,] stds = new float[batchSize , ActionSizes[0]];
        for (int i = 0; i < batchSize; ++i)
        {
            for (int j = 0; j < ActionSizes[0]; ++j)
            {
                means[i, j] = desiredMean;
                stds[i, j] = desiredStd;
            }
        }
        inputList.Add(means);
        inputList.Add(stds);

        var loss = PretrainFunction.Call(inputList);
        var result = new float[] { (float)loss[0].eval() };
        return result;
    }





    public override List<Tensor> GetAllModelWeights()
    {
        List<Tensor> result = new List<Tensor>();
        result.AddRange(network.GetWeights());
        if (runningMean != null)
        {
            result.Add(runningMean); result.Add(runningVariance); result.Add(stepCount);
        }
        return result;
    }


    protected Tensor CreateRunninngNormalizer(Tensor vectorInput, int size)
    {
        using (K.name_scope("InputNormalizer"))
        {
            stepCount = K.variable(0, DataType.Float, "NormalizationStep");

            runningMean = K.zeros(new int[] { size }, DataType.Float, "RunningMean");
            float[] initialVariance = new float[size];
            for (int i = 0; i < size; ++i)
            {
                initialVariance[i] = 1;
            }
            runningVariance = K.variable((Array)initialVariance, DataType.Float, "RunningVariance");

            var meanCurrentObs = K.mean(vectorInput, 0);

            var newMean = runningMean + (meanCurrentObs - runningMean) / (stepCount + 1);
            var newVariance = runningVariance + (meanCurrentObs - newMean) * (meanCurrentObs - runningMean);
            var normalized = K.clip((vectorInput - runningMean) / K.sqrt(runningVariance / (stepCount + 1.0f)), -5.0f, 5.0f);
            //var varCurrentObs = K.mean((vectorInput - meanCurrentObs) * (vectorInput - runningMean), 0);
            //var newMean = 0.95f*runningMean + 0.05f* meanCurrentObs;
            //var newVariance = runningVariance + varCurrentObs;
            //var normalized = K.clip((vectorInput - runningMean) / K.sqrt(runningVariance / (stepCount + 1.0f)), -5.0f, 5.0f);
            UpdateNormalizerFunction = K.function(new List<Tensor>() { vectorInput },
                new List<Tensor> { },
                new List<List<Tensor>>() {
                                new List<Tensor>() { K.update(runningMean,newMean) },
                                new List<Tensor>() { K.update(runningVariance,newVariance) },
                                new List<Tensor>(){K.update_add(stepCount,1.0f) },
            }, "UpdateNormalization");

            return normalized;

        }
    }

}