using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using static Current;

public class ModelPPO : MonoBehaviour
{

    public int stateSize = 4;
    public int actionSize = 2;



    private void Awake()
    {
        Initialize();
    }


    public void Initialize()
    {
        
        var inputStateTensor = UnityTFUtils.Input(new int?[] { stateSize })[0];

        //actor network output mean
        var actorDense1 = new Dense(20, new ReLU(), true, kernel_initializer: new GlorotUniform(scale: 0.01f));
        var actorOutput = new Dense(units:actionSize, activation:null, use_bias:true, kernel_initializer: new GlorotUniform(scale: 0.01f));

        var OutputMean = actorOutput.Call(actorDense1.Call(inputStateTensor)[0])[0];

        //actor network output variance
        var log_sigma_sq = K.variable((new Constant(0)).Call(new int[] { actionSize }, DataType.Float),name:"PPO.log_sigma_square");
        var OutputVariance = K.exp(log_sigma_sq);

        //policy output value
        var policyDense1 = new Dense(20, new ReLU(), true, kernel_initializer: new GlorotUniform(scale: 0.01f));
        var policyOutput = new Dense(units: actionSize, activation: null, use_bias: true, kernel_initializer: new GlorotUniform(scale: 0.01f));

        var OutputValue = policyOutput.Call(policyDense1.Call(inputStateTensor)[0])[0];


        //training needed inputs
        var InputAction = UnityTFUtils.Input(new int?[] { actionSize })[0];
        var InputOldProb = UnityTFUtils.Input(new int?[] { actionSize })[0];
        var InputAdvantage = UnityTFUtils.Input(new int?[] { 1 })[0];
        var InputTargetValue = UnityTFUtils.Input(new int?[] { 1 })[0];
        var InputClipEpsilon = K.constant(0.1);
        var InputValuelossWeight = K.constant(1);
        var InputEntropyLossWeight = K.constant(0);

        // action probability from input action
        var temp = K.mul(OutputVariance, 2 * Mathf.PI * 2.7182818285);
        temp = K.mul(temp, 0.5);
        var OutputEntropy = K.sum(temp, 0, false);

        var actionProb = K.normal_probability(InputAction, OutputMean, OutputVariance);

        // value loss
        var OutputValueLoss = new MeanSquareError().Call(OutputValue, InputTargetValue);

        // Clipped Surrogate loss
        var probRatio = actionProb / (InputOldProb + 0.0000001f);
        var p_opt_a = probRatio * InputAdvantage;
        var p_opt_b = K.clip(probRatio, 1.0f - InputClipEpsilon, 1.0f + InputClipEpsilon) * InputAdvantage;

        var OutputPolicyLoss = 1-K.mean(K.min(p_opt_a, p_opt_b));

        //final weighted loss
        var OutputLoss = OutputPolicyLoss + InputValuelossWeight*OutputValueLoss;
        OutputLoss = OutputLoss - InputEntropyLossWeight*OutputEntropy;
    }

    public float[] EvaluateValue(float[] state)
    {
        throw new System.NotImplementedException();
    }


    public float[] EvaluateAction(float[] state, out float[] actionProbs, SpaceType actionSpace, bool useProbability = true)
    {
        throw new System.NotImplementedException();
    }


    public void TrainBatch(float[] states, float[] actions, float[] actionProbs, float[] targetValues, float[] advantages)
    {
        throw new System.NotImplementedException();
    }
}