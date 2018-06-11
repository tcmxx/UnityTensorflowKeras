using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Accord.Statistics.Distributions.Univariate;
using System;
using System.Linq;
using Accord;
using Accord.Math;
#if UNITY_EDITOR
using UnityEditor;
#endif


using static Current;

public class RLModelPPO : MonoBehaviour
{

    private int stateSize = 4;
    private int actionSize = 2;

    public Function ValueFunction { get; set; }
    public Function ActionFunction { get; set; }
    public Function UpdateFunction { get; set; }

    public Adam optimizer;

    public RLNetworkAC network;

    public bool HasVisualObservation { get; private set; }
    public bool HasVectorObservation { get; private set; }
    public bool HasRecurrent { get; private set; } = false;

    public void Initialize(Brain brain)
    {
        actionSize = brain.brainParameters.vectorActionSize;
        stateSize = brain.brainParameters.vectorObservationSize*brain.brainParameters.numStackedVectorObservations;

        //create basic inputs
        var inputStateTensor = stateSize > 0?UnityTFUtils.Input(new int?[] { stateSize }, name: "InputStates")[0]:null;
        HasVectorObservation = inputStateTensor != null;
        var inputVisualTensors = CreateVisualInputs(brain);
        HasVisualObservation = inputVisualTensors != null;

        //build the network
        Tensor OutputValue = null, OutputMean = null;
        network.BuildNetwork(inputStateTensor, inputVisualTensors, null, null, actionSize, out OutputMean, out OutputValue);

        //actor network output variance
        var log_sigma_sq = K.variable((new Constant(0)).Call(new int[] { actionSize }, DataType.Float), name: "PPO.log_sigma_square");
        var OutputVariance = K.exp(log_sigma_sq);

        //training needed inputs
        var InputAction = UnityTFUtils.Input(new int?[] { actionSize }, name: "InputAction")[0];
        var InputOldProb = UnityTFUtils.Input(new int?[] { actionSize }, name: "InputOldProb")[0];
        var InputAdvantage = UnityTFUtils.Input(new int?[] { 1 }, name: "InputAdvantage")[0];
        var InputTargetValue = UnityTFUtils.Input(new int?[] { 1 }, name: "InputTargetValue")[0];
        var InputClipEpsilon = K.constant(0.1, name: "ClipEpsilon");
        var InputValuelossWeight = K.constant(1, name: "ValueLossWeight");
        var InputEntropyLossWeight = K.constant(0, name: "EntropyLossWeight");

        // action probability from input action
        Tensor OutputEntropy;
        Tensor actionProb;
        using (K.name_scope("ActionProb"))
        {
            var temp = K.mul(OutputVariance, 2 * Mathf.PI * 2.7182818285);
            temp = K.mul(temp, 0.5);
            OutputEntropy = K.sum(temp, 0, false, name: "OutputEntropy");
            actionProb = K.normal_probability(InputAction, OutputMean, OutputVariance);
        }
        // value loss
        var OutputValueLoss = K.mean(new MeanSquareError().Call(OutputValue, InputTargetValue));

        // Clipped Surrogate loss
        Tensor OutputPolicyLoss;
        using (K.name_scope("ClippedCurreogateLoss"))
        {
            var probRatio = actionProb / (InputOldProb + 0.0000001f);
            var p_opt_a = probRatio * InputAdvantage;
            var p_opt_b = K.clip(probRatio, 1.0f - InputClipEpsilon, 1.0f + InputClipEpsilon) * InputAdvantage;

            OutputPolicyLoss = K.mean(1 - K.mean(K.min(p_opt_a, p_opt_b)), name: "ClippedCurreogateLoss");
        }
        //final weighted loss
        var OutputLoss = OutputPolicyLoss + InputValuelossWeight * OutputValueLoss;
        OutputLoss = OutputLoss - InputEntropyLossWeight * OutputEntropy;


        //add inputs, outputs and parameters to the list
        List<Tensor> updateParameters = new List<Tensor>();
        List<Tensor> allInputs = new List<Tensor>();
        List<Tensor> observationInputs = new List<Tensor>();
        updateParameters.AddRange(network.GetWeights());
        updateParameters.Add(log_sigma_sq);

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
        allInputs.Add(InputAction);
        allInputs.Add(InputOldProb);
        allInputs.Add(InputTargetValue);
        allInputs.Add(InputAdvantage);

        //create optimizer and create necessary functions
        optimizer = new Adam(lr: 0.001);
        var updates = optimizer.get_updates(updateParameters, null, OutputLoss); ;
        UpdateFunction = K.function(allInputs, new List<Tensor> { OutputLoss, OutputValueLoss, OutputPolicyLoss }, updates, "UpdateFunction");
        ValueFunction = K.function(observationInputs, new List<Tensor> { OutputValue }, null, "ValueFunction");
        ActionFunction = K.function(observationInputs, new List<Tensor> { OutputMean, OutputVariance }, null, "ActionFunction");
        //test
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
    /// <param name="vectorObservation">current states. Can be batch input</param>
    /// <returns>values of current states</returns>
    public float[] EvaluateValue(float[] vectorObservation, List<float[,,,]> visualObservation)
    {
        List<Array> inputLists = new List<Array>();
        if (HasVectorObservation)
        {
            inputLists.Add(vectorObservation);
        }
        if (HasVisualObservation)
        {
            inputLists.AddRange(visualObservation);
        }

        var result = ValueFunction.Call(inputLists);
        //return new float[] { ((float[,])result[0].eval())[0,0] };
        return ((float[,])result[0].eval()).Flatten();
    }

    /// <summary>
    /// Query actions based on curren states
    /// </summary>
    /// <param name="vectorObservation">current vector states. Can be batch input</param>
    /// <param name="actionProbs">output actions' probabilities</param>
    /// <param name="actionSpace">action space type.</param>
    /// <param name="useProbability">when true, the output actions are sampled based on output mean and variance. Otherwise it uses mean directly.</param>
    /// <returns></returns>
    public float[] EvaluateAction(float[] vectorObservation, out float[] actionProbs, List<float[,,,]> visualObservation, SpaceType actionSpace, bool useProbability = true)
    {
        List<Array> inputLists = new List<Array>();
        if (HasVectorObservation)
        {
            inputLists.Add(vectorObservation);
        }
        if (HasVisualObservation)
        {
            inputLists.AddRange(visualObservation);
        }

        var result = ActionFunction.Call(inputLists);

        var means = ((float[,])result[0].eval()).Flatten();
        var vars = (float[])result[1].eval();

        float[] actions = new float[means.Length];
        actionProbs = new float[means.Length];
        for (int j = 0; j < means.Length; ++j)
        {
            var std = Mathf.Sqrt(vars[j%actionSize]);
            var dis = new NormalDistribution(means[j], std);

            if (useProbability)
                actions[j] = (float)dis.Generate();
            else
                actions[j] = means[j];
            actionProbs[j] = (float)dis.ProbabilityDensityFunction(actions[j]);
        }

        return actions;

    }

    public void SetLearningRate(float rl)
    {
        optimizer.SetLearningRate(rl);
    }

    public float[] TrainBatch(float[] states, float[] actions, float[] actionProbs, float[] targetValues, float[] advantages)
    {
        var loss = UpdateFunction.Call(new List<Array> { states, actions, actionProbs, targetValues, advantages });
        return new float[] { (float)loss[0].eval(), (float)loss[1].eval(), (float)loss[2].eval() };
    }

    
}