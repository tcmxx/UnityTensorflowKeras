using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;
using KerasSharp.Engine.Topology;
using KerasSharp.Backends;
using KerasSharp;
using KerasSharp.Initializers;
using KerasSharp.Activations;
using System;

[CreateAssetMenu()]
public class RLNetowrkACHierarchy : ScriptableObject
{
    public int inLowlevelLayers = 1;
    public int inLowlevelWidth = 128;

    public int actorHighlevelLayers = 1;
    public int actorHighlevelWidth = 128;
    public int valueHighlevelOutLayers = 1;
    public int valueHighlevelOutWidth = 128;


    public int actorOutLowlevelLayers = 1;
    public int actorOutLowlevelWidth = 128;

    public float lowLevelWeightsInitialScale = 0.07f;
    public float highLevelWeightsInitialScale = 0.07f;

    protected List<Tensor> weightsLowlevel;
    protected List<Tensor> weightsHighLevel;

    //the variable for variance
    protected Tensor logSigmaSq = null;


    public void BuildNetwork(Tensor inVectorstateLowlevel, Tensor inVectorstateHighlevel,int outActionSize, SpaceType actionSpace,
        out Tensor outAction, out Tensor outValue, out Tensor outVariance)
    {

        weightsLowlevel = new List<Tensor>();
        weightsHighLevel = new List<Tensor>();
        

        //lowlevel encoder
        var lowlevelEncoder = CreateContinuousStateEncoder(inVectorstateLowlevel, inLowlevelWidth, inLowlevelLayers, "LowlevelEncoder", lowLevelWeightsInitialScale);
        Tensor  encodedLowlevel = lowlevelEncoder.Item1;
        weightsLowlevel.AddRange(lowlevelEncoder.Item2);

        //highlevel 
        Tensor concatedStates = null;
        if (inVectorstateHighlevel != null)
            concatedStates = Current.K.concat(new List<Tensor>() { encodedLowlevel, inVectorstateHighlevel }, 1);
        else
            concatedStates = encodedLowlevel;

        var highlevelEncoder = CreateContinuousStateEncoder(concatedStates, actorHighlevelWidth, actorHighlevelLayers, "ActorHighevelEncoder", highLevelWeightsInitialScale);
        Tensor outputHighlevel = highlevelEncoder.Item1;
        weightsHighLevel.AddRange(highlevelEncoder.Item2);

        //lowlevel actor output
        var actorFinal = CreateContinuousStateEncoder(outputHighlevel, actorOutLowlevelWidth, actorOutLowlevelLayers, "ActorLowlevelOut", lowLevelWeightsInitialScale);
        Tensor encodedAllActor = actorFinal.Item1;
        weightsLowlevel.AddRange(actorFinal.Item2);

        //highlevel value output
        var valueFinal = CreateContinuousStateEncoder(encodedLowlevel, valueHighlevelOutWidth, valueHighlevelOutLayers, "ValueHighlevelOut", highLevelWeightsInitialScale);
        Tensor encodedAllCritic = valueFinal.Item1;
        weightsHighLevel.AddRange(valueFinal.Item2);

        //outputs
        using (Current.K.name_scope("ActorOutput"))
        {
            var actorOutput = new Dense(units: outActionSize, activation: null, use_bias: true, kernel_initializer: new GlorotUniform(scale: lowLevelWeightsInitialScale));
            outAction = actorOutput.Call(encodedAllActor)[0];
            if (actionSpace == SpaceType.discrete)
            {
                outAction = Current.K.softmax(outAction);
            }

            weightsLowlevel.AddRange(actorOutput.weights);
        }

        using (Current.K.name_scope("CriticOutput"))
        {
            var criticOutput = new Dense(units: 1, activation: null, use_bias: true, kernel_initializer: new GlorotUniform(scale: highLevelWeightsInitialScale));
            outValue = criticOutput.Call(encodedAllCritic)[0];
            weightsHighLevel.AddRange(criticOutput.weights);
        }
        //variance
        //actor network output variance
        if (actionSpace == SpaceType.continuous)
        {
            using (Current.K.name_scope("ActorVarianceOutput"))
            {
                logSigmaSq = Current.K.variable((new Constant(0)).Call(new int[] { outActionSize }, DataType.Float), name: "PPO.log_sigma_square");
                outVariance = Current.K.exp(logSigmaSq);
                weightsLowlevel.Add(logSigmaSq);
            }
        }
        else
        {
            outVariance = null;
        }
    }


    protected ValueTuple<Tensor,List<Tensor>> CreateContinuousStateEncoder(Tensor state, int hiddenSize, int numLayers, string scope, float initScale)
    {
        var hidden = state;
        List<Tensor> tempWeights = new List<Tensor>();
        using (Current.K.name_scope(scope))
        {
            for (int i = 0; i < numLayers; ++i)
            {
                var layer = new Dense(hiddenSize, new ReLU(), true, kernel_initializer: new GlorotUniform(scale: initScale));
                hidden = layer.Call(hidden)[0];
                tempWeights.AddRange(layer.weights);
            }
        }

        return ValueTuple.Create(hidden,tempWeights);
    }


    public List<Tensor> GetHighLevelWeights()
    {
        return weightsHighLevel;
    }

    public List<Tensor> GetLowLevelWeights()
    {
        return weightsLowlevel;
    }
    
}
