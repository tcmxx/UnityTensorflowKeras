using KerasSharp;
using KerasSharp.Backends;
using KerasSharp.Engine.Topology;
using KerasSharp.Initializers;
using MLAgents;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu(menuName = "ml-agent/ppo-cma/RLNetworkACSeperateVar")]
public class RLNetworkACSeperateVar : RLNetworkSimpleAC
{
    public bool useSoftclipForMean = false;
    public float maxMean = 1;
    public float minMean = -1;
    protected List<Tensor> actorVarWeights;

    public override void BuildNetworkForContinuousActionSapce(Tensor inVectorObs, List<Tensor> inVisualObs, Tensor inMemery, Tensor inPrevAction, int outActionSize,
        out Tensor outActionMean, out Tensor outValue, out Tensor outActionLogVariance)
    {

        Debug.Assert(inMemery == null, "Currently recurrent input is not supported by RLNetworkSimpleAC");
        Debug.Assert(inPrevAction == null, "Currently previous action input is not supported by RLNetworkSimpleAC");
        Debug.Assert(!(inVectorObs == null && inVisualObs == null), "Network need at least one vector observation or visual observation");
        criticWeights = new List<Tensor>();
        actorWeights = new List<Tensor>();
        actorVarWeights = new List<Tensor>();


        var actorMeanEncoded = CreateObservationStream(inVectorObs, actorHiddenLayers, inVisualObs, inMemery, inPrevAction, "ActorMean");
        var actorVarEncoded = CreateObservationStream(inVectorObs, actorHiddenLayers, inVisualObs, inMemery, inPrevAction, "ActorVar");
        var criticEncoded = CreateObservationStream(inVectorObs, criticHiddenLayers, inVisualObs, inMemery, inPrevAction, "Critic");

        actorWeights.AddRange(actorMeanEncoded.Item2);
        actorVarWeights.AddRange(actorVarEncoded.Item2);
        criticWeights.AddRange(criticEncoded.Item2);

        //concat all inputs
        Tensor encodedAllActorMean = actorMeanEncoded.Item1;
        Tensor encodedAllActorVar = actorVarEncoded.Item1;
        Tensor encodedAllCritic = criticEncoded.Item1;
        

        //outputs
        //mean
        var actorOutputMean = new Dense(units: outActionSize, activation: null, use_bias: actorOutputLayerBias, kernel_initializer: new VarianceScaling(scale: actorOutputLayerInitialScale));
        outActionMean = actorOutputMean.Call(encodedAllActorMean)[0];
        if (useSoftclipForMean)
        {
            outActionMean = SoftClip(outActionMean, minMean, maxMean);
        }
        actorWeights.AddRange(actorOutputMean.weights);

        //var
        var actorOutputVar = new Dense(units: outActionSize, activation: null, use_bias: actorOutputLayerBias, kernel_initializer: new VarianceScaling(scale: actorOutputLayerInitialScale));
        outActionLogVariance = actorOutputVar.Call(encodedAllActorVar)[0];
        //outActionLogVariance = Current.K.exp(outActionLogVariance);
        actorVarWeights.AddRange(actorOutputVar.weights);
        
        //critic
        var criticOutput = new Dense(units: 1, activation: null, use_bias: criticOutputLayerBias, kernel_initializer: new GlorotUniform(scale: criticOutputLayerInitialScale));
        outValue = criticOutput.Call(encodedAllCritic)[0];
        criticWeights.AddRange(criticOutput.weights);

    }

    public override void BuildNetworkForDiscreteActionSpace(Tensor inVectorObs, List<Tensor> inVisualObs, Tensor inMemery, Tensor inPrevAction, int[] outActionSizes, out Tensor[] outActionLogits, out Tensor outValue)
    {
        Debug.LogError("RLNetworkACSeperateVar does not support discrete action space.");
        outValue = null;
        outActionLogits = null;
    }



    Tensor SoftClip(Tensor x, float min, float max)
    {
        return min + (max - min) * Current.K.sigmoid(x);
    }

    public List<Tensor> GetActorMeanWeights()
    {
        return actorWeights;
    }

    public List<Tensor> GetActorVarianceWeights()
    {
        return actorVarWeights;
    }

    public override List<Tensor> GetActorWeights()
    {
        List<Tensor> result = new List<Tensor>();
        result.AddRange(actorWeights);
        result.AddRange(actorVarWeights);
        return result;
    }

    public override List<Tensor> GetWeights()
    {
        List<Tensor> result = new List<Tensor>();
        result.AddRange(actorWeights);
        result.AddRange(actorVarWeights);
        result.AddRange(criticWeights);
        return result;
    }
}
