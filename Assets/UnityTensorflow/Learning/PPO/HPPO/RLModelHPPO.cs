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


public interface IRLModelHPPO
{
    /// <summary>
    /// The entropy loss weight
    /// </summary>
    float EntropyLossWeight { get; set; }

    /// <summary>
    /// The value loss weight
    /// </summary>
    float ValueLossWeight { get; set; }

    /// <summary>
    /// The clip epsilon for PPO actor loss
    /// </summary>
    float ClipEpsilon { get; set; }

    /// <summary>
    /// The value loss clip
    /// </summary>
    float ClipValueLoss { get; set; }

    /// <summary>
    /// Evaluate the values of current states
    /// </summary>
    /// <param name="vectorObservation">Batched vector observations.</param>
    /// <param name="visualObservation">List of batched visual observations.</param>
    /// <returns>Values of the input batched states</returns>
    float[] EvaluateValue(float[,] vectorObservation, List<float[,,,]> visualObservation);

    /// <summary>
    /// Evaluate the desired actions of current states
    /// </summary>
    /// <param name="vectorObservation">Batched vector observations.</param>
    /// <param name="actionProbs">Output action probabilities of the output actions. Used for PPO training.</param>
    /// <param name="visualObservation">List of batched visual observations.</param>
    /// <param name="actionsMask">Action masks for discrete action space. Each element in the list is for one branch of the actions. Can be null if no mask.</param>
    /// <returns>The desired actions of the batched input states.</returns>
    float[,] EvaluateAction(float[,] vectorObservation, out float[,] actionProbs, List<float[,,,]> visualObservation, List<float[,]> actionsMask = null);

    /// <summary>
    /// Evaluate the input actions' probabilities of current states
    /// </summary>
    /// <param name="vectorObservation">Batched vector observations.</param>
    /// <param name="actions">The batched actions that need the probabilies</param>
    /// <param name="visualObservation">List of batched visual observations.</param>
    /// <param name="actionsMask">Action masks for discrete action space. Each element in the list is for one branch of the actions. Can be null if no mask.</param>
    /// <returns>Output action probabilities of the output actions. Used for PPO training.</returns>
    float[,] EvaluateProbability(float[,] vectorObservation, float[,] actions, List<float[,,,]> visualObservation, List<float[,]> actionsMask = null);

    float[] TrainValue(float[,] vectorObservations, List<float[,,,]> visualObservations, float[] oldValues, float[] targetValues);
    float[] TrainPolicy(float[,] vectorObservations, List<float[,,,]> visualObservations, float[,] actions, float[] advantages, float[,] actionProbs, List<float[,]> actionsMask = null);
}


public class RLModelHPPO : LearningModelBase, IRLModelHPPO, ISupervisedLearningModel
{


    protected Function ValueFunction { get; set; }
    protected Function PolicyFunction { get; set; }

    protected Function UpdateValueFunction { get; set; }
    protected Function UpdatePolicyFunction { get; set; }

    protected Function ActionProbabilityFunction { get; set; }

    protected Function UpdateNormalizerFunction { get; set; }

    protected Function UpdateSLFunction { get; set; }
    protected bool SLHasVar { get; private set; } = false;

    [ShowAllPropertyAttr]
    public RLNetworkAC network;

    public OptimizerCreator policyOptimizer;
    public OptimizerCreator valueOptimizer;
    public OptimizerCreator SLOptimizer;

    public bool useInputNormalization = false;
    public float EntropyLossWeight { get; set; }
    public float ValueLossWeight { get; set; }
    public float ClipEpsilon { get; set; }
    public float ClipValueLoss { get; set; }

    //the variables for normalization
    protected Tensor runningMean = null;
    protected Tensor runningVariance = null;
    protected Tensor stepCount = null;

    /// <summary>
    /// Initialize the model without training parts
    /// </summary>
    /// <param name="brainParameters"></param>
    public override void InitializeInner(BrainParameters brainParameters, Tensor vecotrObsTensor, List<Tensor> visualTensors, TrainerParams trainerParams)
    {

        //vector observation normalization
        Tensor normalizedVectorObs = vecotrObsTensor;
        if (useInputNormalization && HasVectorObservation)
        {
            normalizedVectorObs = CreateRunninngNormalizer(normalizedVectorObs, StateSize);
        }
        else if (useInputNormalization)
        {
            Debug.LogWarning("useInputNormalization is turned off because it is not supported in this case");
            useInputNormalization = false;
        }



        //build all stuff
        if (trainerParams is TrainerParamsPPO)
        {
            if (ActionSpace == SpaceType.continuous)
            {
                InitializePPOStructureContinuousAction(vecotrObsTensor, normalizedVectorObs, visualTensors, trainerParams);
            }
            else if (ActionSpace == SpaceType.discrete)
            {
                InitializePPOStructureDiscreteAction(vecotrObsTensor, normalizedVectorObs, visualTensors, trainerParams);
            }

        }else if(trainerParams is TrainerParamsMimic)
        {
            if (ActionSpace == SpaceType.continuous)
            {
                InitializeSLStructureContinuousAction(vecotrObsTensor, normalizedVectorObs, visualTensors, trainerParams);
            }
            else if (ActionSpace == SpaceType.discrete)
            {
                InitializeSLStructureDiscreteAction(vecotrObsTensor, normalizedVectorObs, visualTensors, trainerParams);
            }
        }
    }

    protected void InitializePPOStructureDiscreteAction(Tensor vectorObs, Tensor normalizedVectorObs, List<Tensor> visualObs, TrainerParams trainerParams)
    {

        //all inputs list
        List<Tensor> allObservationInputs = new List<Tensor>();
        if (HasVectorObservation)
        {
            allObservationInputs.Add(vectorObs);
        }
        if (HasVisualObservation)
        {
            allObservationInputs.AddRange(visualObs);
        }

        Tensor[] outputActionsLogits = null; Tensor outputValue = null;
        network.BuildNetworkForDiscreteActionSpace(normalizedVectorObs, visualObs, null, null, ActionSizes, out outputActionsLogits, out outputValue);

        ValueFunction = K.function(allObservationInputs, new List<Tensor> { outputValue }, null, "ValueFunction");

        //the action masks input placeholders
        List<Tensor> actionMasksInputs = new List<Tensor>();
        for (int i = 0; i < ActionSizes.Length; ++i)
        {
            actionMasksInputs.Add(UnityTFUtils.Input(new int?[] { ActionSizes[i] }, name: "AcionMask" + i)[0]);
        }

        Tensor[] outputActions, outputNormalizedLogits;
        CreateDiscreteActionMaskingLayer(outputActionsLogits, actionMasksInputs.ToArray(), out outputActions, out outputNormalizedLogits);

        //output tensors for discrete actions. Includes all action selected actions and the normalized logits of all actions
        var outputDiscreteActions = new List<Tensor>();
        outputDiscreteActions.Add(K.identity(K.cast(ActionSizes.Length == 1 ? outputActions[0] : K.concat(outputActions.ToList(), 1), DataType.Float), "OutputAction"));
        outputDiscreteActions.AddRange(outputNormalizedLogits);
        var actionFunctionInputs = new List<Tensor>();
        actionFunctionInputs.AddRange(allObservationInputs); actionFunctionInputs.AddRange(actionMasksInputs);
        PolicyFunction = K.function(actionFunctionInputs, outputDiscreteActions, null, "ActionFunction");


        TrainerParamsPPO trainingParams = trainerParams as TrainerParamsPPO;

        if (trainingParams != null)
        {
            // action probability from input action
            Tensor outputEntropy;
            List<Tensor> inputActionsDiscreteSeperated = null, onehotInputActions = null;    //for discrete action space

            Tensor inputAction = UnityTFUtils.Input(new int?[] { ActionSizes.Length }, name: "InputActions", dtype: DataType.Int32)[0];

            //split the input for each discrete branch
            var splits = new int[ActionSizes.Length];
            for (int i = 0; i < splits.Length; ++i)
            {
                splits[i] = 1;
            }
            inputActionsDiscreteSeperated = K.split(inputAction, K.constant(splits, dtype: DataType.Int32), K.constant(1, dtype: DataType.Int32), ActionSizes.Length);

            Tensor actionLogProb = null;
            using (K.name_scope("ActionProbAndEntropy"))
            {

                onehotInputActions = inputActionsDiscreteSeperated.Select((x, i) => K.reshape(K.one_hot(x, K.constant<int>(ActionSizes[i], dtype: DataType.Int32), K.constant(1.0f), K.constant(0.0f)), new int[] { -1, ActionSizes[i] })).ToList();

                //entropy
                var entropies = outputActionsLogits.Select((t) => { return K.mean((-1.0f) * K.sum(K.softmax(t) * K.log(K.softmax(t) + 0.00000001f), axis: 1), 0); });
                outputEntropy = entropies.Aggregate((x, y) => { return x + y; });

                //probabilities
                var actionProbsArray = ActionSizes.Select((x, i) => { return K.sum(outputNormalizedLogits[i] * onehotInputActions[i], 1, true); }).ToList();
                //actionLogProb = K.reshape(K.sum(K.log(outputActionFromNetwork) * onehotInputAction, 1), new int[] { -1, 1 });
                actionLogProb = ActionSizes.Length == 1 ? actionProbsArray[0] : K.concat(actionProbsArray, 1);
            }

            List<Tensor> extraPolicyInputs = new List<Tensor>();
            extraPolicyInputs.AddRange(actionFunctionInputs);
            extraPolicyInputs.Add(inputAction);
            CreatePPOOptimizer(trainingParams, outputEntropy, actionLogProb, outputValue, allObservationInputs, network.GetCriticWeights(), extraPolicyInputs, network.GetActorWeights());

        }
    }


    protected void InitializePPOStructureContinuousAction(Tensor vectorObs, Tensor normalizedVectorObs, List<Tensor> visualObs, TrainerParams trainerParams)
    {

        //all inputs list
        List<Tensor> allObservationInputs = new List<Tensor>();
        if (HasVectorObservation)
        {
            allObservationInputs.Add(vectorObs);
        }
        if (HasVisualObservation)
        {
            allObservationInputs.AddRange(visualObs);
        }

        //build the network
        Tensor outputValue = null; Tensor outputActionMean = null; Tensor outputLogVariance = null;
        network.BuildNetworkForContinuousActionSapce(normalizedVectorObs, visualObs, null, null, ActionSizes[0], out outputActionMean, out outputValue, out outputLogVariance);

        //value function
        ValueFunction = K.function(allObservationInputs, new List<Tensor> { outputValue }, null, "ValueFunction");

        Tensor outputActualAction = null, actionLogProb = null, outputVariance = null;

        //build action sampling
        outputVariance = K.exp(outputLogVariance);
        using (K.name_scope("SampleAction"))
        {
            outputActualAction = K.standard_normal(K.shape(outputActionMean), DataType.Float) * K.sqrt(outputVariance) + outputActionMean;

        }
        using (K.name_scope("ActionProbs"))
        {
            actionLogProb = K.log_normal_probability(K.stop_gradient(outputActualAction), outputActionMean, outputVariance, outputLogVariance);
        }
        //action function
        //ActionFunction = K.function(allObservationInputs, new List<Tensor> { outputActualAction, actionLogProb, outputActionMean }, null, "ActionFunction");
        PolicyFunction = K.function(allObservationInputs, new List<Tensor> { outputActualAction, actionLogProb }, null, "ActionFunction");

        var probInputs = new List<Tensor>(); probInputs.AddRange(allObservationInputs); probInputs.Add(outputActualAction);
        //probability function
        ActionProbabilityFunction = K.function(probInputs, new List<Tensor> { actionLogProb }, null, "ActionProbabilityFunction");

        //training related
        TrainerParamsPPO trainingParams = trainerParams as TrainerParamsPPO;
        if (trainingParams != null)
        {
            Tensor outputEntropy;
            using (K.name_scope("Entropy"))
            {
                var temp = 0.5f * (Mathf.Log(2 * Mathf.PI * 2.7182818285f, 2.7182818285f) + outputLogVariance);
                if (outputLogVariance.shape.Length == 2)
                {
                    outputEntropy = K.mean(K.mean(temp, 0, false), name: "OutputEntropy");
                }
                else
                {
                    outputEntropy = K.mean(temp, 0, false, name: "OutputEntropy");
                }
            }

            List<Tensor> extraInputs = new List<Tensor>();
            extraInputs.AddRange(allObservationInputs);
            extraInputs.Add(outputActualAction);
            CreatePPOOptimizer(trainingParams, outputEntropy, actionLogProb, outputValue, allObservationInputs, network.GetCriticWeights(), extraInputs, network.GetActorWeights());

        }

    }


    protected void CreatePPOOptimizer(TrainerParamsPPO trainingParams, Tensor entropy, Tensor actionLogProb, Tensor outputValueFromNetwork, List<Tensor> extraValueInputTensors, List<Tensor> valueWeights, List<Tensor> extraPolicyInputTensors, List<Tensor> policyWeights)
    {
        ClipEpsilon = trainingParams.clipEpsilon;
        ValueLossWeight = trainingParams.valueLossWeight;
        EntropyLossWeight = trainingParams.entropyLossWeight;
        ClipValueLoss = trainingParams.clipValueLoss;


        var inputOldLogProb = UnityTFUtils.Input(new int?[] { ActionSpace == SpaceType.continuous ? ActionSizes[0] : ActionSizes.Length }, name: "InputOldLogProb")[0];
        var inputAdvantage = UnityTFUtils.Input(new int?[] { 1 }, name: "InputAdvantage")[0];
        var inputTargetValue = UnityTFUtils.Input(new int?[] { 1 }, name: "InputTargetValue")[0];
        var inputOldValue = UnityTFUtils.Input(new int?[] { 1 }, name: "InputOldValue")[0];

        var inputClipEpsilon = UnityTFUtils.Input(batch_shape: new int?[] { }, name: "ClipEpsilon", dtype: DataType.Float)[0];
        var inputClipValueLoss = UnityTFUtils.Input(batch_shape: new int?[] { }, name: "ClipValueLoss", dtype: DataType.Float)[0];
        var inputValuelossWeight = UnityTFUtils.Input(batch_shape: new int?[] { }, name: "ValueLossWeight", dtype: DataType.Float)[0];
        var inputEntropyLossWeight = UnityTFUtils.Input(batch_shape: new int?[] { }, name: "EntropyLossWeight", dtype: DataType.Float)[0];


        // value loss   
        Tensor outputValueLoss = null;
        using (K.name_scope("ValueLoss"))
        {
            var clippedValueEstimate = inputOldValue + K.clip(outputValueFromNetwork - inputOldValue, 0.0f - inputClipValueLoss, inputClipValueLoss);
            var valueLoss1 = new MeanSquareError().Call(outputValueFromNetwork, inputTargetValue);
            var valueLoss2 = new MeanSquareError().Call(clippedValueEstimate, inputTargetValue);
            outputValueLoss = K.mean(K.maximum(valueLoss1, valueLoss2));
        }
        //var outputValueLoss = K.mean(valueLoss1);

        // Clipped Surrogate loss
        Tensor outputPolicyLoss;
        using (K.name_scope("ClippedCurreogateLoss"))
        {
            //Debug.LogWarning("testnew");
            //var probStopGradient = K.stop_gradient(actionProb);
            var probRatio = K.exp(actionLogProb - inputOldLogProb);
            var p_opt_a = probRatio * inputAdvantage;
            var p_opt_b = K.clip(probRatio, 1.0f - inputClipEpsilon, 1.0f + inputClipEpsilon) * inputAdvantage;

            outputPolicyLoss = (-1f) * K.mean(K.mean(K.minimun(p_opt_a, p_opt_b)), name: "ClippedCurreogateLoss");
        }

        //final weighted loss
        var finalValueLoss = inputValuelossWeight * outputValueLoss;
        var finalPolicyLoss = outputPolicyLoss - inputEntropyLossWeight * entropy;


        //add inputs, outputs and parameters to the list
        List<Tensor> allPolicyInputs = new List<Tensor>();
        List<Tensor> allValueInputs = new List<Tensor>();
        allPolicyInputs.Add(inputOldLogProb);
        allPolicyInputs.Add(inputAdvantage);
        allPolicyInputs.Add(inputClipEpsilon);
        allPolicyInputs.Add(inputEntropyLossWeight);

        allValueInputs.Add(inputClipValueLoss);
        allValueInputs.Add(inputValuelossWeight);
        allValueInputs.Add(inputTargetValue);
        allValueInputs.Add(inputOldValue);


        allPolicyInputs.AddRange(extraPolicyInputTensors);
        allValueInputs.AddRange(extraValueInputTensors);

        //create optimizer and create necessary functions
        var policyUpdates = AddOptimizer(policyWeights, finalPolicyLoss, policyOptimizer);
        var valueUpdates = AddOptimizer(valueWeights, finalValueLoss, valueOptimizer);

        UpdatePolicyFunction = K.function(allPolicyInputs, new List<Tensor> { finalPolicyLoss, outputPolicyLoss, entropy }, policyUpdates, "UpdatePolicyFunction");
        UpdateValueFunction = K.function(allValueInputs, new List<Tensor> { finalValueLoss }, valueUpdates, "UpdateValueFunction");

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
    /// <param name="actionProbs">output actions' probabilities. note that it is the normalized log probability</param>
    /// <param name="actoinsMask">action mask for discrete action. </param>
    /// <returns></returns>
    public virtual float[,] EvaluateAction(float[,] vectorObservation, out float[,] actionProbs, List<float[,,,]> visualObservation, List<float[,]> actionsMask = null)
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



        float[,] actions = null;
        actionProbs = null;

        if (ActionSpace == SpaceType.continuous)
        {
            var result = PolicyFunction.Call(inputLists);
            actions = ((float[,])result[0].eval());
            actionProbs = ((float[,])result[1].eval());
        }
        else if (ActionSpace == SpaceType.discrete)
        {
            int batchSize = vectorObservation != null ? vectorObservation.GetLength(0) : visualObservation[0].GetLength(0);
            int branchSize = ActionSizes.Length;
            List<float[,]> masks = actionsMask;

            //create all 1 mask if the input mask is null.
            if (masks == null)
            {
                masks = CreateDummyMasks(ActionSizes, batchSize);
            }

            inputLists.AddRange(masks);

            var result = PolicyFunction.Call(inputLists);
            actions = ((float[,])result[0].eval());

            //get the log probabilities
            actionProbs = new float[batchSize, branchSize];
            for (int b = 0; b < branchSize; ++b)
            {
                var tempProbs = ((float[,])result[b + 1].eval());
                int actSize = ActionSizes[b];
                for (int i = 0; i < batchSize; ++i)
                {
                    actionProbs[i, b] = tempProbs[i, Mathf.RoundToInt(actions[i, b])];

                }
            }
        }

        //normlaized the input observations in every calll of eval action
        if (useInputNormalization && HasVectorObservation)
        {
            UpdateNormalizerFunction.Call(new List<Array>() { vectorObservation });
        }

        return actions;
    }


    /// <summary>
    /// Query actions' probabilities based on curren states. The first dimension of the array must be batch dimension. Note that it is the normalized log probability
    /// </summary>
    public virtual float[,] EvaluateProbability(float[,] vectorObservation, float[,] actions, List<float[,,,]> visualObservation, List<float[,]> actionsMask = null)
    {
        Debug.Assert(TrainingEnabled == true, "The model needs to initalized with Training enabled to use EvaluateProbability()");

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

        var actionProbs = new float[actions.GetLength(0), ActionSpace == SpaceType.continuous ? actions.GetLength(1) : 1];

        if (ActionSpace == SpaceType.continuous)
        {
            inputLists.Add(actions);
            var result = ActionProbabilityFunction.Call(inputLists);
            actionProbs = ((float[,])result[0].eval());
        }
        else if (ActionSpace == SpaceType.discrete)
        {
            List<float[,]> masks = actionsMask;
            int batchSize = vectorObservation.GetLength(0);
            int branchSize = ActionSizes.Length;
            //create all 1 mask if the input mask is null.
            if (masks == null)
            {
                masks = CreateDummyMasks(ActionSizes, batchSize);
            }

            inputLists.AddRange(masks);

            var result = PolicyFunction.Call(inputLists);
            //get the log probabilities
            actionProbs = new float[batchSize, branchSize];
            for (int b = 0; b < branchSize; ++b)
            {
                var tempProbs = ((float[,])result[b + 1].eval());
                int actSize = ActionSizes[b];
                for (int i = 0; i < batchSize; ++i)
                {
                    actionProbs[i, b] = tempProbs[i, Mathf.RoundToInt(actions[i, b])];

                }
            }
        }

        return actionProbs;

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


    public virtual float[] TrainValue(float[,] vectorObservations, List<float[,,,]> visualObservations, float[] oldValues, float[] targetValues)
    {
        Debug.Assert(TrainingEnabled == true, "The model needs to initalized with Training enabled to use TrainBatch()");

        List<Array> inputs = new List<Array>();
        inputs.Add(new float[] { ClipValueLoss });
        inputs.Add(new float[] { ValueLossWeight });
        inputs.Add(targetValues);
        inputs.Add(oldValues);
        if (vectorObservations != null)
            inputs.Add(vectorObservations);
        if (visualObservations != null)
            inputs.AddRange(visualObservations);

        var loss = UpdateValueFunction.Call(inputs);
        var result = new float[] { (float)loss[0].eval() };
        return result;
    }
    public virtual float[] TrainPolicy(float[,] vectorObservations, List<float[,,,]> visualObservations, float[,] actions, float[] advantages, float[,] actionProbs, List<float[,]> actionsMask = null)
    {
        Debug.Assert(TrainingEnabled == true, "The model needs to initalized with Training enabled to use TrainBatch()");

        List<Array> inputs = new List<Array>();
        inputs.Add(actionProbs);
        inputs.Add(advantages);
        inputs.Add(new float[] { ClipEpsilon });
        inputs.Add(new float[] { EntropyLossWeight });

        if (vectorObservations != null)
            inputs.Add(vectorObservations);
        if (visualObservations != null)
            inputs.AddRange(visualObservations);
        if (ActionSpace == SpaceType.continuous)
            inputs.Add(actions);
        else if (ActionSpace == SpaceType.discrete)
        {
            inputs.AddRange(actionsMask);
            int[,] actionsInt = actions.Convert(t => Mathf.RoundToInt(t));
            inputs.Add(actionsInt);
        }

        var loss = UpdatePolicyFunction.Call(inputs);
        var result = new float[] { (float)loss[0].eval(), (float)loss[1].eval(), (float)loss[2].eval() };
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

    #region For SupervisedLearning

    protected void InitializeSLStructureDiscreteAction(Tensor vectorObs, Tensor normalizedVectorObs, List<Tensor> visualObs, TrainerParams trainerParams)
    {

        //all inputs list
        List<Tensor> allObservationInputs = new List<Tensor>();
        if (HasVectorObservation)
        {
            allObservationInputs.Add(vectorObs);
        }
        if (HasVisualObservation)
        {
            allObservationInputs.AddRange(visualObs);
        }

        //build basic network
        Tensor[] outputActionsLogits = null;
        Tensor outputValue = null;
        network.BuildNetworkForDiscreteActionSpace(normalizedVectorObs, visualObs, null, null, ActionSizes, out outputActionsLogits, out outputValue);

        //the action masks input placeholders
        List<Tensor> actionMasksInputs = new List<Tensor>();
        for (int i = 0; i < ActionSizes.Length; ++i)
        {
            actionMasksInputs.Add(UnityTFUtils.Input(new int?[] { ActionSizes[i] }, name: "AcionMask" + i)[0]);
        }
        //masking and normalized and get the final action tensor
        Tensor[] outputActions, outputNormalizedLogits;
        CreateDiscreteActionMaskingLayer(outputActionsLogits, actionMasksInputs.ToArray(), out outputActions, out outputNormalizedLogits);

        //output tensors for discrete actions. Includes all action selected actions
        var outputDiscreteActions = new List<Tensor>();
        outputDiscreteActions.Add(K.identity(K.cast(ActionSizes.Length == 1 ? outputActions[0] : K.concat(outputActions.ToList(), 1), DataType.Float), "OutputAction"));
        var actionFunctionInputs = new List<Tensor>();
        actionFunctionInputs.AddRange(allObservationInputs);
        actionFunctionInputs.AddRange(actionMasksInputs);
        PolicyFunction = K.function(actionFunctionInputs, outputDiscreteActions, null, "ActionFunction");


        //build the parts for training
        TrainerParamsMimic trainingParams = trainerParams as TrainerParamsMimic;
        if (trainerParams != null && trainingParams == null)
        {
            Debug.LogError("Trainer params for Supervised learning mode needs to be a TrainerParamsMimic type");
        }
        if (trainingParams != null)
        {
            //training inputs
            var inputActionLabels = UnityTFUtils.Input(new int?[] { ActionSizes.Length }, name: "InputAction", dtype: DataType.Int32)[0];
            //split the input for each discrete branch
            List<Tensor> inputActionsDiscreteSeperated = null, onehotInputActions = null;    //for discrete action space
            var splits = new int[ActionSizes.Length];
            for (int i = 0; i < splits.Length; ++i)
            {
                splits[i] = 1;
            }
            inputActionsDiscreteSeperated = K.split(inputActionLabels, K.constant(splits, dtype: DataType.Int32), K.constant(1, dtype: DataType.Int32), ActionSizes.Length);

            //creat the loss
            onehotInputActions = inputActionsDiscreteSeperated.Select((x, i) => K.reshape(K.one_hot(x, K.constant<int>(ActionSizes[i], dtype: DataType.Int32), K.constant(1.0f), K.constant(0.0f)), new int[] { -1, ActionSizes[i] })).ToList();

            var losses = onehotInputActions.Select((x, i) => K.mean(K.categorical_crossentropy(x, outputNormalizedLogits[i], true))).ToList();
            Tensor loss = losses.Aggregate((x, s) => x + s);

            //add inputs, outputs and parameters to the list
            List<Tensor> updateParameters = network.GetActorWeights();
            List<Tensor> allInputs = new List<Tensor>();
            allInputs.AddRange(actionFunctionInputs);
            allInputs.Add(inputActionLabels);

            //create optimizer and create necessary functions
            var updates = AddOptimizer(updateParameters, loss, SLOptimizer);
            UpdateSLFunction = K.function(allInputs, new List<Tensor> { loss }, updates, "UpdateFunction");
        }
    }



    protected void InitializeSLStructureContinuousAction(Tensor vectorObs, Tensor normalizedVectorObs, List<Tensor> visualObs, TrainerParams trainerParams)
    {
        //build the network
        Tensor outputValue = null; Tensor outputActionMean = null; Tensor outputLogVariance = null;
        network.BuildNetworkForContinuousActionSapce(normalizedVectorObs, visualObs, null, null, ActionSizes[0], out outputActionMean, out outputValue, out outputLogVariance);
        Tensor outputAction = outputActionMean;
        Tensor outputVar = K.exp(outputLogVariance);
        SLHasVar = outputLogVariance != null;

        List<Tensor> observationInputs = new List<Tensor>();
        if (HasVectorObservation)
        {
            observationInputs.Add(vectorObs);
        }
        if (HasVisualObservation)
        {
            observationInputs.AddRange(visualObs);
        }
        if (SLHasVar)
            PolicyFunction = K.function(observationInputs, new List<Tensor> { outputAction, outputVar }, null, "ActionFunction");
        else
            PolicyFunction = K.function(observationInputs, new List<Tensor> { outputAction }, null, "ActionFunction");

        //build the parts for training
        TrainerParamsMimic trainingParams = trainerParams as TrainerParamsMimic;
        if (trainerParams != null && trainingParams == null)
        {
            Debug.LogError("Trainer params for Supervised learning mode needs to be a TrainerParamsMimic type");
        }
        if (trainingParams != null)
        {
            //training inputs
            var inputActionLabel = UnityTFUtils.Input(new int?[] { ActionSizes[0] }, name: "InputAction", dtype: DataType.Float)[0];
            //creat the loss
            Tensor loss = null;
            if (SLHasVar)
            {
                loss = K.mean(K.mean(0.5 * K.square(inputActionLabel - outputAction) / outputVar + 0.5 * outputLogVariance));
            }
            else
                loss = K.mean(new MeanSquareError().Call(inputActionLabel, outputAction));

            //add inputs, outputs and parameters to the list
            List<Tensor> updateParameters = network.GetActorWeights();
            List<Tensor> allInputs = new List<Tensor>();
            allInputs.AddRange(observationInputs);
            allInputs.Add(inputActionLabel);

            //create optimizer and create necessary functions
            var updates = AddOptimizer(updateParameters, loss, SLOptimizer);
            UpdateSLFunction = K.function(allInputs, new List<Tensor> { loss }, updates, "UpdateFunction");
        }
    }





    /// <summary>
    /// THis is implemented for ISupervisedLearingModel so that this model can also be used for TrainerMimic
    /// </summary>
    /// <param name="vectorObservation"></param>
    /// <param name="visualObservation"></param>
    /// <returns>(mean, var) var will be null for discrete</returns>
    public ValueTuple<float[,], float[,]> EvaluateAction(float[,] vectorObservation, List<float[,,,]> visualObservation, List<float[,]> actionsMask)
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

        if (ActionSpace == SpaceType.discrete)
        {
            int batchSize = vectorObservation != null ? vectorObservation.GetLength(0) : visualObservation[0].GetLength(0);
            int branchSize = ActionSizes.Length;
            List<float[,]> masks = actionsMask;
            //create all 1 mask if the input mask is null.
            if (masks == null)
            {
                masks = CreateDummyMasks(ActionSizes, batchSize);
            }
            inputLists.AddRange(masks);
        }

        var result = PolicyFunction.Call(inputLists);

        float[,] actions = ((float[,])result[0].eval());

        float[,] outputVar = null;
        if (SLHasVar)
        {
            object varTemp = result[1].eval();
            if (varTemp is float[,])
                outputVar = (float[,])result[1].eval();
            else
                outputVar = null;
        }

        //normlaized the input observations in every calll of eval action
        if (useInputNormalization && HasVectorObservation)
        {
            UpdateNormalizerFunction.Call(new List<Array>() { vectorObservation });
        }


        return ValueTuple.Create(actions, outputVar);
    }
    /// <summary>
    /// Training for supervised learning
    /// </summary>
    /// <param name="vectorObservations"></param>
    /// <param name="visualObservations"></param>
    /// <param name="actions"></param>
    /// <returns></returns>
    public float TrainBatch(float[,] vectorObservations, List<float[,,,]> visualObservations, float[,] actions, List<float[,]> actionsMask = null)
    {
        Debug.Assert(TrainingEnabled == true, "The model needs to initalized with Training enabled to use TrainBatch()");

        List<Array> inputs = new List<Array>();
        if (vectorObservations != null)
            inputs.Add(vectorObservations);
        if (visualObservations != null)
            inputs.AddRange(visualObservations);



        if (ActionSpace == SpaceType.continuous)
            inputs.Add(actions);
        else if (ActionSpace == SpaceType.discrete)
        {
            List<float[,]> masks = actionsMask;
            int batchSize = actions.GetLength(0);
            //create all 1 mask if the input mask is null.
            if (masks == null)
            {
                masks = CreateDummyMasks(ActionSizes, batchSize);
            }
            inputs.AddRange(masks);

            int[,] actionsInt = actions.Convert(t => Mathf.RoundToInt(t));
            inputs.Add(actionsInt);
        }

        var loss = UpdateSLFunction.Call(inputs);
        var result = (float)loss[0].eval();

        return result;
    }
    #endregion
}
