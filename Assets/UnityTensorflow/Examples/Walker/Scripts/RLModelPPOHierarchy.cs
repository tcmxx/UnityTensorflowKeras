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
            var splited = K.split(stateTensor, K.constant(new int[] { lowLevelObservationSize, highLevelObservationSize }, dtype: DataType.Int32), K.constant(1, dtype: DataType.Int32),2);
            inputLowLevelTensor = splited[0];
            inputHighLevelTensor = splited[1];
        }
        else
        {
            inputLowLevelTensor = stateTensor;
        }

        Tensor outputValue = null; Tensor outputAction = null; Tensor outputVariance = null;
        //build the network
        networkHierarchy.BuildNetwork(inputLowLevelTensor, inputHighLevelTensor, ActionSize, ActionSpace, out outputAction, out outputValue,out outputVariance);



        List<Tensor> allobservationInputs = new List<Tensor>();
        if (HasVectorObservation)
        {
            allobservationInputs.Add(stateTensor);
        }
        if (HasVisualObservation)
        {
            allobservationInputs.AddRange(visualTensors);
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
                    outputEntropy = K.mean(temp, 0, false, name: "OutputEntropy");
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
            var outputValueLoss = K.mean(new MeanSquareError().Call(outputValue, inputTargetValue));

            // Clipped Surrogate loss
            Tensor outputPolicyLoss;
            using (K.name_scope("ClippedCurreogateLoss"))
            {
                var probRatio = actionProb / (inputOldProb + 0.0000001f);
                var p_opt_a = probRatio * inputAdvantage;
                var p_opt_b = K.clip(probRatio, 1.0f - inputClipEpsilon, 1.0f + inputClipEpsilon) * inputAdvantage;

                outputPolicyLoss = K.mean(1 - K.mean(K.min(p_opt_a, p_opt_b)), name: "ClippedCurreogateLoss");
            }
            //final weighted loss
            var outputLoss = outputPolicyLoss + inputValuelossWeight * outputValueLoss;
            outputLoss = outputLoss - inputEntropyLossWeight * outputEntropy;


            //add inputs, outputs and parameters to the list
            List<Tensor> updateParameters = GetAllModelWeights();
            List<Tensor> allInputs = new List<Tensor>();
            
            allInputs.Add(stateTensor);
            allInputs.Add(inputAction);
            allInputs.Add(inputOldProb);
            allInputs.Add(inputTargetValue);
            allInputs.Add(inputAdvantage);
            allInputs.Add(inputClipEpsilon);
            allInputs.Add(inputValuelossWeight);
            allInputs.Add(inputEntropyLossWeight);

            //create optimizer and create necessary functions
            var updates = AddOptimizer(updateParameters, outputLoss, optimizer);
            UpdateFunction = K.function(allInputs, new List<Tensor> { outputLoss, outputValueLoss, outputPolicyLoss }, updates, "UpdateFunction");
        }


    }

    /// <summary>
    /// evaluate the value of current states
    /// </summary>
    /// <param name="vectorObservation">current vector observation. The first dimension of the array is the batch dimension.</param>
    /// <param name="visualObservation">current visual observation. The first dimension of the array is the batch dimension.</param>
    /// <returns>values of current states</returns>
    public override float[] EvaluateValue(float[,] vectorObservation, List<float[,,,]> visualObservation)
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
    /// <param name="actionProbs">output actions' probabilities</param>
    /// <param name="useProbability">when true, the output actions are sampled based on output mean and variance. Otherwise it uses mean directly.</param>
    /// <returns></returns>
    public override float[,] EvaluateAction(float[,] vectorObservation, out float[,] actionProbs, List<float[,,,]> visualObservation, bool useProbability = true)
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
        var vars = ActionSpace == SpaceType.continuous ? (float[])result[1].eval() : null;

        float[,] actions = new float[outputAction.GetLength(0), ActionSpace == SpaceType.continuous ? outputAction.GetLength(1) : 1];
        actionProbs = new float[outputAction.GetLength(0), ActionSpace == SpaceType.continuous ? outputAction.GetLength(1) : 1];

        if (ActionSpace == SpaceType.continuous)
        {
            for (int j = 0; j < outputAction.GetLength(0); ++j)
            {
                for (int i = 0; i < outputAction.GetLength(1); ++i)
                {
                    try
                    {
                        var std = Mathf.Sqrt(vars[i]);
                        var dis = new NormalDistribution(outputAction[j, i], std);

                        if (useProbability)
                            actions[j, i] = (float)dis.Generate();
                        else
                            actions[j, i] = outputAction[j, i];
                        actionProbs[j, i] = (float)dis.ProbabilityDensityFunction(actions[j, i]);
                    }catch(Exception e)
                    {
                        //Debug.LogWarning("NaN action from neural network detected. Force it to 0.");
                        actions[j, i] = 0;
                        actionProbs[j, i] = 1;
                    }
                }
            }
        }
        else if (ActionSpace == SpaceType.discrete)
        {
            for (int j = 0; j < outputAction.GetLength(0); ++j)
            {
                if (useProbability)
                    actions[j, 0] = MathUtils.IndexByChance(outputAction.GetRow(j));
                else
                    actions[j, 0] = outputAction.GetRow(j).ArgMax();

                actionProbs[j, 0] = outputAction.GetRow(j)[Mathf.RoundToInt(actions[j, 0])];
            }
        }

        return actions;

    }


    /// <summary>
    /// Query actions' probabilities based on curren states. The first dimension of the array must be batch dimension
    /// </summary>
    public override float[,] EvaluateProbability(float[,] vectorObservation, float[,] actions, List<float[,,,]> visualObservation)
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

        var result = ActionFunction.Call(inputLists);

        var outputAction = ((float[,])result[0].eval());
        var vars = ActionSpace == SpaceType.continuous ? (float[])result[1].eval() : null;

        var actionProbs = new float[outputAction.GetLength(0), ActionSpace == SpaceType.continuous ? outputAction.GetLength(1) : 1];

        if (ActionSpace == SpaceType.continuous)
        {
            for (int j = 0; j < outputAction.GetLength(0); ++j)
            {
                for (int i = 0; i < outputAction.GetLength(1); ++i)
                {
                    var std = Mathf.Sqrt(vars[i]);
                    if (outputAction[j, i] == float.NaN || std == float.NaN || actions[j, i] == float.NaN)
                    {
                        actionProbs[j, i] = 0.5f;
                        Debug.LogWarning("not valid output action mean:" + outputAction[j, i] + " or std:" + std +" or action to evaluate: " + actions[j, i]);
                        continue;
                    }

                    var dis = new NormalDistribution(outputAction[j, i], std);
                    
                    actionProbs[j, i] = (float)dis.ProbabilityDensityFunction(actions[j, i]);
                }
            }
        }
        else if (ActionSpace == SpaceType.discrete)
        {
            for (int j = 0; j < outputAction.GetLength(0); ++j)
            {
                actionProbs[j, 0] = outputAction.GetRow(j)[Mathf.RoundToInt(actions[j, 0])];
            }
        }

        return actionProbs;

    }



    public override float[] TrainBatch(float[,] vectorObservations, List<float[,,,]> visualObservations, float[,] actions, float[,] actionProbs, float[] targetValues, float[] advantages)
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

        inputs.Add(actionProbs);
        inputs.Add(targetValues);
        inputs.Add(advantages);
        inputs.Add(new float[] { ClipEpsilon });
        inputs.Add(new float[] { ValueLossWeight });
        inputs.Add(new float[] { EntropyLossWeight });

        var loss = UpdateFunction.Call(inputs);
        var result = new float[] { (float)loss[0].eval(), (float)loss[1].eval(), (float)loss[2].eval() };

        return result;
        //Debug.LogWarning("test save graph");
        //((UnityTFBackend)K).ExportGraphDef("SavedGraph/PPOTest.pb");
        //return new float[] { 0, 0, 0 }; //test for memeory allocation
    }

    public override List<Tensor> GetAllModelWeights()
    {
        return networkHierarchy.GetHighLevelWeights();
    }


}
