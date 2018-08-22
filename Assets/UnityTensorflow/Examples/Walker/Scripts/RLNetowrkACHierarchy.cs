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
public class RLNetowrkACHierarchy : UnityNetwork
{
    public List<SimpleDenseLayerDef> inLowlevelLayers;

    public List<SimpleDenseLayerDef> actorHighlevelLayers;
    public List<SimpleDenseLayerDef> valueHighlevelLayers;


    public List<SimpleDenseLayerDef> actorLowlevelLayers;

    public float actorOutputLayerInitialScale = 0.1f;
    public bool actorOutputLayerBias = true;

    public float criticOutputLayerInitialScale = 0.1f;
    public bool criticOutputLayerBias = true;


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
        var lowlevelEncoder = BuildSequentialLayers(inLowlevelLayers, inVectorstateLowlevel, "LowlevelEncoder");
        Tensor  encodedLowlevel = lowlevelEncoder.Item1;
        weightsLowlevel.AddRange(lowlevelEncoder.Item2);



        //highlevel 
        Tensor concatedStates = null;
        if (inVectorstateHighlevel != null)
            concatedStates = Current.K.concat(new List<Tensor>() { encodedLowlevel, inVectorstateHighlevel }, 1);
        else
            concatedStates = encodedLowlevel;
        
        var highlevelEncoder = BuildSequentialLayers( actorHighlevelLayers, concatedStates, "ActorHighevelEncoder");
        Tensor outputHighlevel = highlevelEncoder.Item1;
        weightsHighLevel.AddRange(highlevelEncoder.Item2);

        //lowlevel actor output
        var actorFinal = BuildSequentialLayers(actorLowlevelLayers, outputHighlevel, "ActorLowlevelOut");
        Tensor encodedAllActor = actorFinal.Item1;
        weightsLowlevel.AddRange(actorFinal.Item2);

        //highlevel value output
        var valueFinal = BuildSequentialLayers(valueHighlevelLayers, concatedStates, "ValueHighlevelOut");
        Tensor encodedAllCritic = valueFinal.Item1;
        weightsHighLevel.AddRange(valueFinal.Item2);

        //outputs
        using (Current.K.name_scope("ActorOutput"))
        {
            var actorOutput = new Dense(units: outActionSize, activation: null, use_bias: actorOutputLayerBias, kernel_initializer: new VarianceScaling(scale: actorOutputLayerInitialScale));
            outAction = actorOutput.Call(encodedAllActor)[0];
            if (actionSpace == SpaceType.discrete)
            {
                outAction = Current.K.softmax(outAction);
            }

            weightsLowlevel.AddRange(actorOutput.weights);
        }

        using (Current.K.name_scope("CriticOutput"))
        {
            var criticOutput = new Dense(units: 1, activation: null, use_bias: criticOutputLayerBias, kernel_initializer: new GlorotUniform(scale: criticOutputLayerInitialScale));
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
                weightsHighLevel.Add(logSigmaSq);
            }
        }
        else
        {
            outVariance = null;
        }
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
