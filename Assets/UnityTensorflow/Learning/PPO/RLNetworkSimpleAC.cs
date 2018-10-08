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
using System.Linq;

using static KerasSharp.Backends.Current;

[CreateAssetMenu(menuName = "ml-agent/ppo/RLNetworkSimpleAC")]
public class RLNetworkSimpleAC : RLNetworkAC
{

    public List<SimpleDenseLayerDef> actorHiddenLayers;
    public List<SimpleDenseLayerDef> criticHiddenLayers;

    public float actorOutputLayerInitialScale = 0.01f;
    public bool actorOutputLayerBias = true;

    public float criticOutputLayerInitialScale = 1f;
    public bool criticOutputLayerBias = true;

    public float visualEncoderInitialScale = 1f;
    public bool visualEncoderBias = true;

    [Tooltip("If shared, actor and critic will use the share the network weights.criticHiddenLayers will be ignored. ")]
    public bool shareEncoder = false;

    protected List<Tensor> criticWeights;
    protected List<Tensor> actorWeights;

    public override void BuildNetworkForContinuousActionSapce(Tensor inVectorObs, List<Tensor> inVisualObs, Tensor inMemery, Tensor inPrevAction, int outActionSize,
        out Tensor outActionMean, out Tensor outValue, out Tensor outActionLogVariance)
    {

        Tensor encodedAllActor = null;
        CreateCommonLayers(inVectorObs, inVisualObs, inMemery, inPrevAction, out outValue, out encodedAllActor, shareEncoder);

        //outputs
        var actorOutput = new Dense(units: outActionSize, activation: null, use_bias: actorOutputLayerBias, kernel_initializer: new VarianceScaling(scale: actorOutputLayerInitialScale));
        outActionMean = actorOutput.Call(encodedAllActor)[0];
        actorWeights.AddRange(actorOutput.weights);

        var logSigmaSq = Current.K.variable((new Constant(0)).Call(new int[] { outActionSize }, DataType.Float), name: "PPO.log_sigma_square");
        outActionLogVariance = logSigmaSq;
        actorWeights.Add(logSigmaSq);


    }

    public override void BuildNetworkForDiscreteActionSpace(Tensor inVectorObs, List<Tensor> inVisualObs, Tensor inMemery, Tensor inPrevAction,int[] outActionSizes, out Tensor[] outActionLogits, out Tensor outValue)
    {
        Tensor encodedAllActor = null;
        CreateCommonLayers(inVectorObs, inVisualObs, inMemery, inPrevAction, out outValue, out encodedAllActor, shareEncoder);

        List<Tensor> policy_branches = new List<Tensor>();
        foreach(var size in outActionSizes)
        {
            var tempOutput = new Dense(units: size, activation: null, use_bias: actorOutputLayerBias, kernel_initializer: new VarianceScaling(scale: actorOutputLayerInitialScale));
            policy_branches.Add( tempOutput.Call(encodedAllActor)[0]);
            actorWeights.AddRange(tempOutput.weights);
        }
        outActionLogits = policy_branches.ToArray();
    }

    /// <summary>
    /// Create the layers that are common for discrete and continuous action space
    /// </summary>
    /// <param name="inVectorObs"></param>
    /// <param name="inVisualObs"></param>
    /// <param name="inMemery"></param>
    /// <param name="inPrevAction"></param>
    /// <param name="outValue"></param>
    /// <param name="encodedAllActor"></param>
    /// <param name="shareEncoder"></param>
    protected void CreateCommonLayers(Tensor inVectorObs, List<Tensor> inVisualObs, Tensor inMemery, Tensor inPrevAction, out Tensor outValue, out Tensor encodedAllActor, bool shareEncoder = false)
    {

        actorWeights = new List<Tensor>();
        criticWeights = new List<Tensor>();

        ValueTuple<Tensor, List<Tensor>> actorEncoded, criticEncoded;
        if (!shareEncoder)
        {
            actorEncoded = CreateObservationStream(inVectorObs, actorHiddenLayers, inVisualObs, inMemery, inPrevAction, "Actor");
            criticEncoded = CreateObservationStream(inVectorObs, criticHiddenLayers, inVisualObs, inMemery, inPrevAction, "Critic");
        }
        else
        {
            actorEncoded = CreateObservationStream(inVectorObs, actorHiddenLayers, inVisualObs, inMemery, inPrevAction, "ActorCritic");
            criticEncoded = actorEncoded;
        }

        actorWeights.AddRange(actorEncoded.Item2);
        criticWeights.AddRange(criticEncoded.Item2);

        var criticOutput = new Dense(units: 1, activation: null, use_bias: criticOutputLayerBias, kernel_initializer: new GlorotUniform(scale: criticOutputLayerInitialScale));
        outValue = criticOutput.Call(criticEncoded.Item1)[0];
        criticWeights.AddRange(criticOutput.weights);

        encodedAllActor = actorEncoded.Item1;
    }


    protected ValueTuple<Tensor, List<Tensor>> CreateObservationStream(Tensor inVectorObs, List<SimpleDenseLayerDef> layerDefs, List<Tensor> inVisualObs, Tensor inMemery, Tensor inPrevAction, string encoderName)
    {
        Debug.Assert(inMemery == null, "Currently recurrent input is not supported by RLNetworkSimpleAC");
        Debug.Assert(inPrevAction == null, "Currently previous action input is not supported by RLNetworkSimpleAC");
        Debug.Assert(!(inVectorObs == null && inVisualObs == null), "Network need at least one vector observation or visual observation");

        List<Tensor>  allWeights = new List<Tensor>();
        Tensor encodedAll = null;
        //visual encoders
        Tensor encodedVisual = null;
        if (inVisualObs != null)
        {
            List<Tensor> visualEncoded = new List<Tensor>();
            foreach (var v in inVisualObs)
            {
                var ha = CreateVisualEncoder(v, layerDefs, encoderName + "VisualEncoder");

                allWeights.AddRange(ha.Item2);
                visualEncoded.Add(ha.Item1);
            }
            if (inVisualObs.Count > 1)
            {
                //Debug.LogError("Tensorflow does not have gradient for concat operation in C yet. Please only use one observation.");
                encodedVisual = Current.K.stack(visualEncoded, 1);
                encodedVisual = Current.K.batch_flatten(encodedVisual);
            }
            else
            {
                encodedVisual = visualEncoded[0];
            }
        }

        //vector states encode
        Tensor encodedVectorState = null;
        if (inVectorObs != null)
        {
            var output = BuildSequentialLayers(layerDefs, inVectorObs, encoderName + "StateEncoder");
            encodedVectorState = output.Item1;
            allWeights.AddRange(output.Item2);
        }

        //concat all inputs
        if (inVisualObs == null && inVectorObs != null)
        {
            encodedAll = encodedVectorState;
        }
        else if (inVisualObs != null && inVectorObs == null)
        {
            encodedAll = encodedVisual;
        }
        else if (inVisualObs != null && inVectorObs != null)
        {
            //Debug.LogWarning("Tensorflow does not have gradient for concat operation in C yet. Please only use one type of observation if you need training.");
            encodedAll = Current.K.concat(new List<Tensor>() { encodedVectorState, encodedVisual }, 1);
        }

        return ValueTuple.Create(encodedAll, allWeights);
    }








    protected ValueTuple<Tensor, List<Tensor>> CreateVisualEncoder(Tensor visualInput, List<SimpleDenseLayerDef> denseLayers, string scope)
    {
        //use the same encoder as in UnityML's python codes
        Tensor temp;
        List<Tensor> returnWeights = new List<Tensor>();
        using (Current.K.name_scope(scope))
        {
            var conv1 = new Conv2D(16, new int[] { 8, 8 }, new int[] { 4, 4 }, use_bias: visualEncoderBias, kernel_initializer: new GlorotUniform(scale: visualEncoderInitialScale), activation: new ELU());
            var conv2 = new Conv2D(32, new int[] { 4, 4 }, new int[] { 2, 2 }, use_bias: visualEncoderBias, kernel_initializer: new GlorotUniform(scale: visualEncoderInitialScale), activation: new ELU());

            temp = conv1.Call(visualInput)[0];
            temp = conv2.Call(temp)[0];

            var flatten = new Flatten();
            //temp = Current.K.batch_flatten(temp);
            temp = flatten.Call(temp)[0];
            returnWeights.AddRange(conv1.weights);
            returnWeights.AddRange(conv2.weights);
        }

        var output = BuildSequentialLayers(denseLayers, temp, scope);
        var hiddenFlat = output.Item1;
        returnWeights.AddRange(output.Item2);


        return ValueTuple.Create(hiddenFlat, returnWeights);
    }



    public override List<Tensor> GetWeights()
    {
        var result = new List<Tensor>();
        result.AddRange(actorWeights);
        result.AddRange(criticWeights.Where(x=>!actorWeights.Contains(x)));
        return result;
    }

    public override List<Tensor> GetActorWeights()
    {
        return actorWeights;
    }

    public override List<Tensor> GetCriticWeights()
    {
        return criticWeights;
    }


}
