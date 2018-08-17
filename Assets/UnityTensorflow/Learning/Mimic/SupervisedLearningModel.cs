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
    /// 
    /// </summary>
    /// <param name="vectorObservation"></param>
    /// <param name="visualObservation"></param>
    /// <returns>(means,vars). If the supervised learning model does not support var, the second can be null</returns>
    ValueTuple<float[,], float[,]> EvaluateAction(float[,] vectorObservation, List<float[,,,]> visualObservation);
    float TrainBatch(float[,] vectorObservations, List<float[,,,]> visualObservations, float[,] actions);
    bool HasVariance();
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


    public override void InitializeInner(BrainParameters brainParameters, Tensor inputStateTensor, List<Tensor> inputVisualTensors,  TrainerParams trainerParams)
    {
        //build the network
        var networkOutputs = network.BuildNetwork(inputStateTensor, inputVisualTensors, null, ActionSize,ActionSpace);
        Tensor outputAction = networkOutputs.Item1;
        Tensor outputVar = networkOutputs.Item2;
        hasVariance = outputVar != null && brainParameters.vectorActionSpaceType == SpaceType.continuous;

        List<Tensor> observationInputs = new List<Tensor>();
        if (HasVectorObservation)
        {
            observationInputs.Add(inputStateTensor);
        }
        if (HasVisualObservation)
        {
            observationInputs.AddRange(inputVisualTensors);
        }
        if(hasVariance)
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
            var inputActionLabel = UnityTFUtils.Input(new int?[] { ActionSpace == SpaceType.continuous ? ActionSize : 1 }, name: "InputAction", dtype: ActionSpace == SpaceType.continuous ? DataType.Float : DataType.Int32)[0];
            //creat the loss
            Tensor loss = null;
            if (ActionSpace == SpaceType.discrete)
            {
                Tensor actionOnehot = K.one_hot(inputActionLabel, K.constant(ActionSize, dtype: DataType.Int32), K.constant(1.0f), K.constant(0.0f));
                Tensor reshapedOnehot = K.reshape(actionOnehot, new int[] { -1, ActionSize });
                loss = K.mean(K.categorical_crossentropy(reshapedOnehot, outputAction, false));
            }
            else
            {
                if (hasVariance)
                {
                    loss =K.mean( K.mean(0.5 * K.square(inputActionLabel - outputAction) / outputVar + 0.5 * K.log(outputVar)));
                }
                else
                    loss = K.mean(new MeanSquareError().Call(inputActionLabel, outputAction));
            }
            //add inputs, outputs and parameters to the list
            List<Tensor> updateParameters = network.GetWeights();
            List<Tensor> allInputs = new List<Tensor>();


            if (HasVectorObservation)
            {
                allInputs.Add(inputStateTensor);
                observationInputs.Add(inputStateTensor);
            }
            if (HasVisualObservation)
            {
                allInputs.AddRange(inputVisualTensors);
                observationInputs.AddRange(inputVisualTensors);
            }
            allInputs.Add(inputActionLabel);

            //create optimizer and create necessary functions
            var updates = AddOptimizer(updateParameters, loss, optimizer);
            UpdateFunction = K.function(allInputs, new List<Tensor> { loss }, updates, "UpdateFunction");
        }
        
    }


    public bool HasVariance()
    {
        return hasVariance;
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="vectorObservation"></param>
    /// <param name="visualObservation"></param>
    /// <returns></returns>
    public virtual ValueTuple<float[,], float[,]> EvaluateAction(float[,] vectorObservation, List<float[,,,]> visualObservation)
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
        
        float[,] actions = new float[outputAction.GetLength(0), ActionSpace == SpaceType.continuous ? outputAction.GetLength(1) : 1];
        if (ActionSpace == SpaceType.continuous)
        {
            for (int j = 0; j < outputAction.GetLength(0); ++j)
            {
                for (int i = 0; i < outputAction.GetLength(1); ++i)
                {
                    actions[j, i] = outputAction[j, i];
                }
            }
        }
        else if (ActionSpace == SpaceType.discrete)
        {
            for (int j = 0; j < outputAction.GetLength(0); ++j)
            {
                actions[j, 0] = outputAction.GetRow(j).ArgMax();
            }
        }

        float[,] outputVar = null;
        if (hasVariance)
        {
            outputVar = (float[,])result[1].eval();
        }

        return ValueTuple.Create(actions, outputVar) ;
    }


    public virtual float TrainBatch(float[,] vectorObservations, List<float[,,,]> visualObservations, float[,] actions)
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

    float[,] INeuralEvolutionModel.EvaluateAction(float[,] vectorObservation, List<float[,,,]> visualObservation)
    {
        return EvaluateAction(vectorObservation, visualObservation).Item1;
    }

    public List<Tensor> GetWeightsForNeuralEvolution()
    {
        return network.GetWeights();
    }
}
