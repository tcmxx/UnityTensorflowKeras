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
public class RLNetworkSimpleAC : RLNetworkAC
{

    public List<SimpleDenseLayerDef> actorHiddenLayers;
    public List<SimpleDenseLayerDef> criticHiddenLayers;

    public float actorOutputLayerInitialScale = 0.01f;
    public bool actorOutputLayerBias = true;

    public float criticOutputLayerInitialScale = 1f;
    public bool criticOutputLayerBias = true;

    public float visualEncoderInitialScale = 0.01f;
    public bool visualEncoderBias = true;
    
    protected List<Tensor> criticWeights;
    protected List<Tensor> actorWeights;

    public override void BuildNetwork(Tensor inVectorstate, List<Tensor> inVisualState, Tensor inMemery, Tensor inPrevAction, int outActionSize, SpaceType actionSpace,
        out Tensor outAction, out Tensor outValue, out Tensor outVariance)
    {

        Debug.Assert(inMemery == null, "Currently recurrent input is not supported by RLNetworkSimpleAC");
        Debug.Assert(inPrevAction == null, "Currently previous action input is not supported by RLNetworkSimpleAC");
        Debug.Assert(!(inVectorstate == null && inVisualState == null), "Network need at least one vector observation or visual observation");
        //Debug.Assert(actionSpace == SpaceType.continuous, "Only continuous action space is supported by RLNetworkSimpleAC");
        criticWeights = new List<Tensor>();
        actorWeights = new List<Tensor>();

        //visual encoders
        Tensor encodedVisualActor = null;
        Tensor encodedVisualCritic = null;
        if (inVisualState != null)
        {
            List<Tensor> visualEncodedActor = new List<Tensor>();
            List<Tensor> visualEncodedCritic = new List<Tensor>();
            foreach (var v in inVisualState)
            {
                var ha = CreateVisualEncoder(v, actorHiddenLayers, "ActorVisualEncoder");
                var hc = CreateVisualEncoder(v, criticHiddenLayers, "CriticVisualEncoder");

                actorWeights.AddRange(ha.Item2);
                visualEncodedActor.Add(ha.Item1);

                criticWeights.AddRange(hc.Item2);
                visualEncodedCritic.Add(hc.Item1);
            }
            if(inVisualState.Count > 1)
            {
                //Debug.LogError("Tensorflow does not have gradient for concat operation in C yet. Please only use one observation.");
                encodedVisualActor = Current.K.stack(visualEncodedActor, 1);
                encodedVisualActor = Current.K.batch_flatten(encodedVisualActor);
                encodedVisualCritic = Current.K.stack(visualEncodedCritic, 1);
                encodedVisualCritic = Current.K.batch_flatten(encodedVisualCritic);
            }
            else
            {
                encodedVisualActor = visualEncodedActor[0];
                encodedVisualCritic = visualEncodedCritic[0];
            }


        }

        

        //vector states encode
        Tensor encodedVectorStateActor = null;
        Tensor encodedVectorStateCritic = null;
        if (inVectorstate != null)
        {
            var output = BuildSequentialLayers(actorHiddenLayers, inVectorstate, "ActorStateEncoder");
            encodedVectorStateActor = output.Item1;
            actorWeights.AddRange(output.Item2);
            output = BuildSequentialLayers(criticHiddenLayers, inVectorstate, "CriticStateEncoder");
            encodedVectorStateCritic = output.Item1;
            criticWeights.AddRange(output.Item2);
        }

        //concat all inputs
        Tensor encodedAllActor = null;
        Tensor encodedAllCritic = null;

        if(inVisualState == null && inVectorstate != null)
        {
            encodedAllActor = encodedVectorStateActor;
            encodedAllCritic = encodedVectorStateCritic;
        }
        else if(inVisualState != null && inVectorstate == null)
        {
            encodedAllActor = encodedVisualActor;
            encodedAllCritic = encodedVisualCritic;
        }
        else if(inVisualState != null && inVectorstate != null)
        {
            //Debug.LogWarning("Tensorflow does not have gradient for concat operation in C yet. Please only use one type of observation if you need training.");
            encodedAllActor = Current.K.concat(new List<Tensor>() { encodedVectorStateActor,encodedVisualActor},1);
            encodedAllCritic = Current.K.concat(new List<Tensor>() { encodedVectorStateCritic, encodedVisualCritic }, 1);
        }


        //outputs
        var actorOutput = new Dense(units: outActionSize, activation: null, use_bias: actorOutputLayerBias, kernel_initializer: new VarianceScaling(scale: actorOutputLayerInitialScale));
        outAction = actorOutput.Call(encodedAllActor)[0];
        if (actionSpace == SpaceType.discrete)
        {
            outAction = Current.K.softmax(outAction);
        }
        actorWeights.AddRange(actorOutput.weights);

        var criticOutput = new Dense(units: 1, activation: null, use_bias: criticOutputLayerBias, kernel_initializer: new GlorotUniform(scale: criticOutputLayerInitialScale));
        outValue = criticOutput.Call(encodedAllCritic)[0];
        criticWeights.AddRange(criticOutput.weights);

        //output variance. Currently not depending on the inputs for this simple network implementation        
        if (actionSpace == SpaceType.continuous)
        {
            var logSigmaSq = Current.K.variable((new Constant(0)).Call(new int[] { outActionSize }, DataType.Float), name: "PPO.log_sigma_square");
            outVariance = Current.K.exp(logSigmaSq);
            actorWeights.Add(logSigmaSq);
        }
        else
        {
            outVariance = null;
        }
        
    }


    protected ValueTuple<Tensor, List<Tensor>> CreateVisualEncoder(Tensor visualInput, List<SimpleDenseLayerDef> denseLayers, string scope)
    {
        //use the same encoder as in UnityML's python codes
        Tensor temp;
        List<Tensor> returnWeights = new List<Tensor>();
        using (Current.K.name_scope(scope))
        {
            var conv1 = new Conv2D(16, new int[] { 8, 8 }, new int[] { 4, 4 },use_bias:visualEncoderBias ,kernel_initializer: new GlorotUniform(scale: visualEncoderInitialScale), activation: new ELU());
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


        return ValueTuple.Create(hiddenFlat,returnWeights);
    }



    public override List<Tensor> GetWeights()
    {
        var result = new List<Tensor>();
        result.AddRange(actorWeights);
        result.AddRange(criticWeights);
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
