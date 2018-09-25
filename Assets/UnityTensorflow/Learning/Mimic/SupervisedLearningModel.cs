using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using Accord.Math;
using System.Linq;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
#if UNITY_EDITOR
using UnityEditor;
#endif
using static KerasSharp.Backends.Current;
using MLAgents;
using KerasSharp.Optimizers;
using KerasSharp.Models;
using KerasSharp.Engine.Topology;
using KerasSharp.Losses;
using KerasSharp;
using KerasSharp.Backends;




public interface ISupervisedLearningModel
{
    /// <summary>
    /// Evaluate the desired actions of current states.
    /// </summary>
    /// <param name="vectorObservation">Batched vector observations.</param>
    /// <param name="visualObservation">List of batched visual observations.</param>
    /// <param name="actionsMask">Action masks for discrete action space. Each element in the list is for one branch of the actions. Can be null if no mask.</param>
    /// <returns>(means,vars). If the supervised learning model does not support var, the second can be null</returns>
    ValueTuple<float[,], float[,]> EvaluateAction(float[,] vectorObservation, List<float[,,,]> visualObservation, List<float[,]> actionsMask = null);

    /// <summary>
    /// Train a batch for Supervised learning
    /// </summary>
    /// <param name="vectorObservation">Batched vector observations.</param>
    /// <param name="visualObservation">List of batched visual observations.</param>
    /// <param name="actions">Desired actions under input states.</param>
    /// <param name="actionsMask">Action masks for discrete action space. Each element in the list is for one branch of the actions. Can be null if no mask.</param>
    /// <returns></returns>
    float TrainBatch(float[,] vectorObservations, List<float[,,,]> visualObservations, float[,] actions, List<float[,]> actionsMask = null);
}

/// <summary>
/// actor critic network abstract class
/// </summary>
public class SupervisedLearningModel : LearningModelBase, ISupervisedLearningModel, INeuralEvolutionModel
{
    [ShowAllPropertyAttr]
    public SupervisedLearningNetwork network;
    public OptimizerCreator optimizer;
    protected Function ActionFunction { get; set; }
    protected Function UpdateFunction { get; set; }

    protected bool hasVariance;


    public override void InitializeInner(BrainParameters brainParameters, Tensor inputStateTensor, List<Tensor> inputVisualTensors, TrainerParams trainerParams)
    {
        //build the network
        if(ActionSpace == SpaceType.continuous)
        {
            InitializeSLStructureContinuousAction(inputStateTensor, inputVisualTensors, trainerParams);
        }
        else if(ActionSpace == SpaceType.discrete)
        {
            InitializeSLStructureDiscreteAction(inputStateTensor, inputVisualTensors, trainerParams);
        }

    }

    protected void InitializeSLStructureDiscreteAction(Tensor vectorObs, List<Tensor> visualObs, TrainerParams trainerParams)
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
        List<Tensor> outputActionsLogits = network.BuildNetworkForDiscreteActionSpace(vectorObs, visualObs, null, ActionSizes);

        //the action masks input placeholders
        List<Tensor> actionMasksInputs = new List<Tensor>();
        for (int i = 0; i < ActionSizes.Length; ++i)
        {
            actionMasksInputs.Add(UnityTFUtils.Input(new int?[] { ActionSizes[i] }, name: "AcionMask" + i)[0]);
        }
        //masking and normalized and get the final action tensor
        Tensor[] outputActions, outputNormalizedLogits;
        CreateDiscreteActionMaskingLayer(outputActionsLogits.ToArray(), actionMasksInputs.ToArray(), out outputActions, out outputNormalizedLogits);

        //output tensors for discrete actions. Includes all action selected actions
        var outputDiscreteActions = new List<Tensor>();
        outputDiscreteActions.Add(K.identity(K.cast(ActionSizes.Length == 1 ? outputActions[0] : K.concat(outputActions.ToList(), 1), DataType.Float), "OutputAction"));
        var actionFunctionInputs = new List<Tensor>();
        actionFunctionInputs.AddRange(allObservationInputs);
        actionFunctionInputs.AddRange(actionMasksInputs);
        ActionFunction = K.function(actionFunctionInputs, outputDiscreteActions, null, "ActionFunction");


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
            List<Tensor> updateParameters = network.GetWeights();
            List<Tensor> allInputs = new List<Tensor>();
            allInputs.AddRange(actionFunctionInputs);
            allInputs.Add(inputActionLabels);

            //create optimizer and create necessary functions
            var updates = AddOptimizer(updateParameters, loss, optimizer);
            UpdateFunction = K.function(allInputs, new List<Tensor> { loss }, updates, "UpdateFunction");
        }
    }



    protected void InitializeSLStructureContinuousAction(Tensor vectorObs, List<Tensor> visualObs, TrainerParams trainerParams)
    {
        //build the network
        var networkOutputs = network.BuildNetworkForContinuousActionSapce(vectorObs, visualObs, null, ActionSizes[0]);
        Tensor outputAction = networkOutputs.Item1;
        Tensor outputVar = networkOutputs.Item2;
        hasVariance = outputVar != null;

        List<Tensor> observationInputs = new List<Tensor>();
        if (HasVectorObservation)
        {
            observationInputs.Add(vectorObs);
        }
        if (HasVisualObservation)
        {
            observationInputs.AddRange(visualObs);
        }
        if (hasVariance)
            ActionFunction = K.function(observationInputs, new List<Tensor> { outputAction, outputVar }, null, "ActionFunction");
        else
            ActionFunction = K.function(observationInputs, new List<Tensor> { outputAction }, null, "ActionFunction");

        //build the parts for training
        TrainerParamsMimic trainingParams = trainerParams as TrainerParamsMimic;
        if (trainerParams != null && trainingParams == null)
        {
            Debug.LogError("Trainer params for Supervised learning mode needs to be a TrainerParamsMimic type");
        }
        if (trainingParams != null)
        {
            //training inputs
            var inputActionLabel = UnityTFUtils.Input(new int?[] {  ActionSizes[0]}, name: "InputAction", dtype:  DataType.Float)[0];
            //creat the loss
            Tensor loss = null;
            if (hasVariance)
            {
                loss = K.mean(K.mean(0.5 * K.square(inputActionLabel - outputAction) / outputVar + 0.5 * K.log(outputVar)));
            }
            else
                loss = K.mean(new MeanSquareError().Call(inputActionLabel, outputAction));

            //add inputs, outputs and parameters to the list
            List<Tensor> updateParameters = network.GetWeights();
            List<Tensor> allInputs = new List<Tensor>();
            allInputs.AddRange(observationInputs);
            allInputs.Add(inputActionLabel);

            //create optimizer and create necessary functions
            var updates = AddOptimizer(updateParameters, loss, optimizer);
            UpdateFunction = K.function(allInputs, new List<Tensor> { loss }, updates, "UpdateFunction");
        }
    }







    /// <summary>
    /// 
    /// </summary>
    /// <param name="vectorObservation"></param>
    /// <param name="visualObservation"></param>
    /// <returns></returns>
    public virtual ValueTuple<float[,], float[,]> EvaluateAction(float[,] vectorObservation, List<float[,,,]> visualObservation, List<float[,]> actionsMask = null)
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

        if(ActionSpace == SpaceType.discrete)
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

        var result = ActionFunction.Call(inputLists);

        float[,] actions = ((float[,])result[0].eval());
        
        float[,] outputVar = null;
        if (hasVariance)
        {
            outputVar = (float[,])result[1].eval();
        }

        return ValueTuple.Create(actions, outputVar);
    }


    public virtual float TrainBatch(float[,] vectorObservations, List<float[,,,]> visualObservations, float[,] actions, List<float[,]> actionsMask = null)
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

        var loss = UpdateFunction.Call(inputs);
        var result = (float)loss[0].eval();

        return result;
    }




    public override List<Tensor> GetAllModelWeights()
    {
        return network.GetWeights();
    }

    float[,] INeuralEvolutionModel.EvaluateActionNE(float[,] vectorObservation, List<float[,,,]> visualObservation, List<float[,]> actionsMask)
    {
        return EvaluateAction(vectorObservation, visualObservation, actionsMask).Item1;
    }

    public List<Tensor> GetWeightsForNeuralEvolution()
    {
        return network.GetWeights();
    }
}
