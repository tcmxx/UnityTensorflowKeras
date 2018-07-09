using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Accord.Statistics.Distributions.Univariate;
using System;
using System.Linq;
using Accord;
using Accord.Math;
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

public class RLModelPPO : MonoBehaviour
{

    public int StateSize { get; private set; }
    public int ActionSize { get; private set; }
    public SpaceType ActionSpace { get; private set; }

    public Function ValueFunction { get; private set; }
    public Function ActionFunction { get; private set; }
    public Function UpdateFunction { get; private set; }

    public Adam optimizer;

    public RLNetworkAC network;

    public bool HasVisualObservation { get; private set; }
    public bool HasVectorObservation { get; private set; }
    public bool HasRecurrent { get; private set; } = false;
    
    public float EntropyLossWeight { get; set; }
    public float ValueLossWeight { get; set; }
    public float ClipEpsilon { get; set; }

    //the variable for variance
    protected Tensor logSigmaSq = null;

    

    public virtual void Initialize(Brain brain, TrainerParamsPPO trainingParams)
    {
        ActionSize = brain.brainParameters.vectorActionSize;
        StateSize = brain.brainParameters.vectorObservationSize*brain.brainParameters.numStackedVectorObservations;
        ActionSpace = brain.brainParameters.vectorActionSpaceType;

        //create basic inputs
        var inputStateTensor = StateSize > 0?UnityTFUtils.Input(new int?[] { StateSize }, name: "InputStates")[0]:null;
        HasVectorObservation = inputStateTensor != null;
        var inputVisualTensors = CreateVisualInputs(brain);
        HasVisualObservation = inputVisualTensors != null;

        //build the network
        Tensor outputValue = null, outputAction = null;
        network.BuildNetwork(inputStateTensor, inputVisualTensors, null, null, ActionSize, ActionSpace, out outputAction, out outputValue);

        //actor network output variance
        Tensor outputVariance = null;
        if (ActionSpace == SpaceType.continuous)
        {
            logSigmaSq = K.variable((new Constant(0)).Call(new int[] { ActionSize }, DataType.Float), name: "PPO.log_sigma_square");
            outputVariance = K.exp(logSigmaSq);
        }
        //training needed inputs
        var inputAction = UnityTFUtils.Input(new int?[] { ActionSpace == SpaceType.continuous?ActionSize:1 }, name: "InputAction", dtype:ActionSpace == SpaceType.continuous?DataType.Float:DataType.Int32)[0];
        var inputOldProb = UnityTFUtils.Input(new int?[] { ActionSpace == SpaceType.continuous ? ActionSize : 1 }, name: "InputOldProb")[0];
        var inputAdvantage = UnityTFUtils.Input(new int?[] { 1 }, name: "InputAdvantage")[0];
        var inputTargetValue = UnityTFUtils.Input(new int?[] { 1 }, name: "InputTargetValue")[0];

        ClipEpsilon = trainingParams.clipEpsilon;
        ValueLossWeight = trainingParams.valueLossWeight;
        EntropyLossWeight = trainingParams.entroyLossWeight;

        /*var inputClipEpsilon = K.constant(trainingParams.clipEpsilon, name: "ClipEpsilon");
        var inputValuelossWeight = K.constant(trainingParams.valueLossWeight, name: "ValueLossWeight");
        var inputEntropyLossWeight = K.constant(trainingParams.entroyLossWeight, name: "EntropyLossWeight");*/
        var inputClipEpsilon = UnityTFUtils.Input(batch_shape:new int?[] {  }, name: "ClipEpsilon", dtype: DataType.Float)[0];
        var inputValuelossWeight = UnityTFUtils.Input(batch_shape: new int?[] {  }, name: "ValueLossWeight", dtype: DataType.Float)[0];
        var inputEntropyLossWeight = UnityTFUtils.Input(batch_shape: new int?[] {  }, name: "EntropyLossWeight", dtype: DataType.Float)[0];

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
                var onehotInputAction = K.one_hot(inputAction, K.constant<int>(ActionSize,dtype:DataType.Int32), K.constant(1.0f), K.constant(0.0f));
                onehotInputAction = K.reshape(onehotInputAction, new int[] { -1, ActionSize });
                outputEntropy = K.mean((-1.0f) * K.sum(outputAction * K.log(outputAction + 0.00000001f), axis: 1),0);
                actionProb = K.reshape(K.sum(outputAction* onehotInputAction,1),new int[] { -1,1});
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
        List<Tensor> observationInputs = new List<Tensor>();

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
        allInputs.Add(inputAction);
        allInputs.Add(inputOldProb);
        allInputs.Add(inputTargetValue);
        allInputs.Add(inputAdvantage);
        allInputs.Add(inputClipEpsilon);
        allInputs.Add(inputValuelossWeight);
        allInputs.Add(inputEntropyLossWeight);

        //create optimizer and create necessary functions
        optimizer = new Adam(lr: 0.001);
        var updates = optimizer.get_updates(updateParameters, null, outputLoss); ;
        UpdateFunction = K.function(allInputs, new List<Tensor> { outputLoss, outputValueLoss, outputPolicyLoss }, updates, "UpdateFunction");
        ValueFunction = K.function(observationInputs, new List<Tensor> { outputValue }, null, "ValueFunction");
        if (ActionSpace == SpaceType.continuous)
        {
            ActionFunction = K.function(observationInputs, new List<Tensor> { outputAction, outputVariance }, null, "ActionFunction");
        }
        else
        {
            ActionFunction = K.function(observationInputs, new List<Tensor> { outputAction }, null, "ActionFunction");
        }


        //test
        //Debug.LogWarning("Tensorflow Graph is saved for test purpose at: SavedGraph/PPOTest.pb");
        //((UnityTFBackend)K).ExportGraphDef("SavedGraph/PPOTest.pb");
    }

    protected List<Tensor> CreateVisualInputs(Brain brain)
    {
        if(brain.brainParameters.cameraResolutions == null || brain.brainParameters.cameraResolutions.Length == 0)
        {
            return null;
        }
        List<Tensor> allInputs = new List<Tensor>();
        int i = 0;
        foreach(var r in brain.brainParameters.cameraResolutions)
        {
            int width = r.width;
            int height = r.height;
            int channels;
            if (r.blackAndWhite)
                channels = 1;
            else
                channels = 3;

            var input = UnityTFUtils.Input(new int?[] { height, width, channels }, name: "InputVisual" + i)[0];
            allInputs.Add(input);

            i ++;
        }

        return allInputs;
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
        var value =  ((float[,])result[0].eval()).Flatten();
        return value;
    }

    /// <summary>
    /// Query actions based on curren states. The first dimension of the array must be batch dimension
    /// </summary>
    /// <param name="vectorObservation">current vector states. Can be batch input</param>
    /// <param name="actionProbs">output actions' probabilities</param>
    /// <param name="useProbability">when true, the output actions are sampled based on output mean and variance. Otherwise it uses mean directly.</param>
    /// <returns></returns>
    public virtual float[,] EvaluateAction(float[,] vectorObservation, out float[,] actionProbs, List<float[,,,]> visualObservation, bool useProbability = true)
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
        var vars = ActionSpace == SpaceType.continuous?(float[])result[1].eval():null;
        
        float[,] actions = new float[outputAction.GetLength(0), ActionSpace == SpaceType.continuous?outputAction.GetLength(1):1];
        actionProbs = new float[outputAction.GetLength(0), ActionSpace == SpaceType.continuous ? outputAction.GetLength(1) : 1];

        if (ActionSpace == SpaceType.continuous)
        {
            for (int j = 0; j < outputAction.GetLength(0); ++j)
            {
                for (int i = 0; i < outputAction.GetLength(1); ++i)
                {
                    var std = Mathf.Sqrt(vars[i]);
                    var dis = new NormalDistribution(outputAction[j, i], std);

                    if (useProbability)
                        actions[j, i] = (float)dis.Generate();
                    else
                        actions[j, i] = outputAction[j, i];
                    actionProbs[j, i] = (float)dis.ProbabilityDensityFunction(actions[j, i]);
                }
            }
        }else if(ActionSpace == SpaceType.discrete)
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
    public virtual float[,] EvaluateProbability(float[,] vectorObservation, float[,] actions, List<float[,,,]> visualObservation)
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
        
        var actionProbs = new float[outputAction.GetLength(0), ActionSpace == SpaceType.continuous ? outputAction.GetLength(1) : 1];

        if (ActionSpace == SpaceType.continuous)
        {
            for (int j = 0; j < outputAction.GetLength(0); ++j)
            {
                for (int i = 0; i < outputAction.GetLength(1); ++i)
                {
                    var std = Mathf.Sqrt(vars[i]);
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


    public virtual void SetLearningRate(float rl)
    {
        optimizer.SetLearningRate(rl);
    }

    public virtual float[] TrainBatch(float[,] vectorObservations, List<float[,,,]> visualObservations, float[,] actions, float[,] actionProbs, float[] targetValues, float[] advantages)
    {
        List<Array> inputs = new List<Array>();
        if (vectorObservations != null)
            inputs.Add(vectorObservations);
        if (visualObservations != null)
            inputs.AddRange(visualObservations);
        if(ActionSpace == SpaceType.continuous)
            inputs.Add(actions);
        else if(ActionSpace == SpaceType.discrete)
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
        var result =  new float[] { (float)loss[0].eval(), (float)loss[1].eval(), (float)loss[2].eval() };

        return result;
        //Debug.LogWarning("test save graph");
        //((UnityTFBackend)K).ExportGraphDef("SavedGraph/PPOTest.pb");
        //return new float[] { 0, 0, 0 }; //test for memeory allocation
    }



    /// <summary>
    /// save the models all parameters to a byte array
    /// </summary>
    /// <returns></returns>
    public virtual byte[] SaveCheckpoint()
    {
        List<Array> data = GetAllModelWeights().Select(t => (Array)t.eval()).ToList();
        data.AddRange(GetAllOptimizerWeights());

        List<float[]> flattenedData = new List<float[]>();
        foreach(var d in data)
        {
            flattenedData.Add(d.FlattenAndConvertArray<float>());
        }

        var binFormatter = new BinaryFormatter();
        var mStream = new MemoryStream();
        binFormatter.Serialize(mStream, flattenedData);
        return mStream.ToArray();
    }

    public virtual void RestoreCheckpoint(byte[]  data)
    {
        //deserialize the data
        var mStream = new MemoryStream(data);
        var binFormatter = new BinaryFormatter();
        var floatData = (List<float[]>)binFormatter.Deserialize(mStream);

        List<Array> arrayData = floatData.ConvertAll(t => (Array)t);
        var optimizerWeightLength = GetAllOptimizerWeights().Count;   //used for initialize the graph.
        var modelWeigthLength = GetAllModelWeights().Count;      //get the length of model weights and training param weights
        SetAllModelWeights(arrayData.GetRange(0, modelWeigthLength));

        if (arrayData.Count >= modelWeigthLength + optimizerWeightLength && optimizerWeightLength > 0)
        {
            SetAllOptimizerWeights(arrayData.GetRange(modelWeigthLength, optimizerWeightLength));
        }
    }

    public virtual List<Tensor> GetAllModelWeights()
    {
        List<Tensor> updateParameters = new List<Tensor>();
        updateParameters.AddRange(network.GetWeights());
        if(logSigmaSq != null)
            updateParameters.Add(logSigmaSq);
        return updateParameters;
    }
    public virtual List<Array> GetAllOptimizerWeights()
    {
        return optimizer.get_weights();
    }

    public virtual void SetAllModelWeights(List<Array> values)
    {
        List<Tensor> updateParameters = new List<Tensor>();
        updateParameters.AddRange(network.GetWeights());
        updateParameters.Add(logSigmaSq);

        Debug.Assert(values.Count == updateParameters.Count, "Counts of input values and parameters to update do not match.");

        for(int i = 0; i < updateParameters.Count; ++i)
        {
            Debug.Assert(values[i].GetLength().IsEqual(Mathf.Abs(updateParameters[i].shape.Aggregate((t, s) => t * s).Value)), "Input array shape does not match the Tensor to set value");
            K.set_value(updateParameters[i], values[i]);
        }
    }
    public virtual void SetAllOptimizerWeights(List<Array> values)
    {
        optimizer.set_weights(values);
    }
}