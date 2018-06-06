using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Accord.Statistics.Distributions.Univariate;
using System;
using System.Linq;

using static Current;

public class ModelPPO : MonoBehaviour
{

    public int stateSize = 4;
    public int actionSize = 2;

    public Function ValueFunction { get; set; }
    public Function ActionFunction { get; set; }
    public Function UpdateFunction { get; set; }


    public void Initialize(Brain brain)
    {


        actionSize = brain.brainParameters.vectorActionSize;
        stateSize = brain.brainParameters.vectorObservationSize;

        var inputStateTensor = UnityTFUtils.Input(new int?[] { stateSize }, name: "InputStates")[0];

        //actor network output mean
        var actorDense1 = new Dense(20, new ReLU(), true, kernel_initializer: new GlorotUniform(scale: 0.01f));
        var actorOutput = new Dense(units: actionSize, activation: null, use_bias: true, kernel_initializer: new GlorotUniform(scale: 0.01f));

        var OutputMean = actorOutput.Call(actorDense1.Call(inputStateTensor)[0])[0];




        //actor network output variance
        var log_sigma_sq = K.variable((new Constant(0)).Call(new int[] { actionSize }, DataType.Float), name: "PPO.log_sigma_square");
        var OutputVariance = K.exp(log_sigma_sq);




        //value networkoutput value
        var valueDense1 = new Dense(20, new ReLU(), true, kernel_initializer: new GlorotUniform(scale: 0.01f));
        var valueOutput = new Dense(units: 1, activation: null, use_bias: true, kernel_initializer: new GlorotUniform(scale: 0.01f));

        var OutputValue = valueOutput.Call(valueDense1.Call(inputStateTensor)[0])[0];






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

        updateParameters.AddRange(actorDense1.weights);
        updateParameters.AddRange(actorOutput.weights);
        updateParameters.AddRange(valueDense1.weights);
        updateParameters.AddRange(valueOutput.weights);
        updateParameters.Add(log_sigma_sq);

        allInputs.Add(inputStateTensor);
        allInputs.Add(InputAction);
        allInputs.Add(InputOldProb);
        allInputs.Add(InputTargetValue);
        allInputs.Add(InputAdvantage);

        var optimizer = new Adam();
        var updates = optimizer.get_updates(updateParameters, null, OutputLoss); ;
        UpdateFunction = K.function(allInputs, new List<Tensor> { OutputLoss }, updates, "UpdateFunction");
        ValueFunction = K.function(new List<Tensor> { inputStateTensor }, new List<Tensor> { OutputValue }, null, "ValueFunction");
        ActionFunction = K.function(new List<Tensor> { inputStateTensor }, new List<Tensor> { OutputMean, OutputVariance }, null, "ActionFunction");
        //test
        ((UnityTFBackend)K).ExportGraphDef("SavedGraph/PPOTest.pb");
    }

    public float[] EvaluateValueOne(float[] state)
    {
        var result = ValueFunction.Call(new List<Array> { state });
        return new float[] { ((float[,])result[0].eval())[0,0] };
    }


    public float[] EvaluateActionOne(float[] state, out float[] actionProbs, SpaceType actionSpace, bool useProbability = true)
    {
        var result = ActionFunction.Call(new List<Array> { state });

        var means = (float[,])result[0].eval();
        var vars = (float[])result[1].eval();

        float[] actions = new float[means.Length];
        actionProbs = new float[means.Length];
        for (int j = 0; j < actionSize; ++j)
        {
            var std = Mathf.Sqrt(vars[j]);
            var dis = new NormalDistribution(means[0,j], std);
            
            actions[j] = (float)dis.Generate();
            actionProbs[j] = (float)dis.ProbabilityDensityFunction(actions[j]);
        }

        return actions;

    }


    public float TrainBatch(float[] states, float[] actions, float[] actionProbs, float[] targetValues, float[] advantages)
    {
        var loss = UpdateFunction.Call(new List<Array> { states, actions, actionProbs, targetValues, advantages });
        return (float)loss[0].eval();
    }
}