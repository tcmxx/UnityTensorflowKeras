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
using static KerasSharp.Backends.Current;

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

    public override void BuildNetworkForContinuousActionSapce(Tensor inVectorObs, List<Tensor> inVisualObs, Tensor inMemery, Tensor inPrevAction, int outActionSize,
        out Tensor outActionMean, out Tensor outValue, out Tensor outActionLogVariance)
    {

        Tensor encodedAllActor = null;
        CreateCommonLayers(inVectorObs, inVisualObs, inMemery, inPrevAction, out outValue, out encodedAllActor);

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
        CreateCommonLayers(inVectorObs, inVisualObs, inMemery, inPrevAction, out outValue, out encodedAllActor);

        List<Tensor> policy_branches = new List<Tensor>();
        foreach(var size in outActionSizes)
        {
            var tempOutput = new Dense(units: size, activation: null, use_bias: actorOutputLayerBias, kernel_initializer: new VarianceScaling(scale: actorOutputLayerInitialScale));
            policy_branches.Add( tempOutput.Call(encodedAllActor)[0]);
            actorWeights.AddRange(tempOutput.weights);
        }
        outActionLogits = policy_branches.ToArray();
    }




    protected void CreateCommonLayers(Tensor inVectorObs, List<Tensor> inVisualObs, Tensor inMemery, Tensor inPrevAction, out Tensor outValue, out Tensor encodedAllActor)
    {
        Debug.Assert(inMemery == null, "Currently recurrent input is not supported by RLNetworkSimpleAC");
        Debug.Assert(inPrevAction == null, "Currently previous action input is not supported by RLNetworkSimpleAC");
        Debug.Assert(!(inVectorObs == null && inVisualObs == null), "Network need at least one vector observation or visual observation");

        criticWeights = new List<Tensor>();
        actorWeights = new List<Tensor>();

        //visual encoders
        Tensor encodedVisualActor = null;
        Tensor encodedVisualCritic = null;
        if (inVisualObs != null)
        {
            List<Tensor> visualEncodedActor = new List<Tensor>();
            List<Tensor> visualEncodedCritic = new List<Tensor>();
            foreach (var v in inVisualObs)
            {
                var ha = CreateVisualEncoder(v, actorHiddenLayers, "ActorVisualEncoder");
                var hc = CreateVisualEncoder(v, criticHiddenLayers, "CriticVisualEncoder");

                actorWeights.AddRange(ha.Item2);
                visualEncodedActor.Add(ha.Item1);

                criticWeights.AddRange(hc.Item2);
                visualEncodedCritic.Add(hc.Item1);
            }
            if (inVisualObs.Count > 1)
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
        if (inVectorObs != null)
        {
            var output = BuildSequentialLayers(actorHiddenLayers, inVectorObs, "ActorStateEncoder");
            encodedVectorStateActor = output.Item1;
            actorWeights.AddRange(output.Item2);
            output = BuildSequentialLayers(criticHiddenLayers, inVectorObs, "CriticStateEncoder");
            encodedVectorStateCritic = output.Item1;
            criticWeights.AddRange(output.Item2);
        }

        //concat all inputs
        encodedAllActor = null;
        Tensor encodedAllCritic = null;

        if (inVisualObs == null && inVectorObs != null)
        {
            encodedAllActor = encodedVectorStateActor;
            encodedAllCritic = encodedVectorStateCritic;
        }
        else if (inVisualObs != null && inVectorObs == null)
        {
            encodedAllActor = encodedVisualActor;
            encodedAllCritic = encodedVisualCritic;
        }
        else if (inVisualObs != null && inVectorObs != null)
        {
            //Debug.LogWarning("Tensorflow does not have gradient for concat operation in C yet. Please only use one type of observation if you need training.");
            encodedAllActor = Current.K.concat(new List<Tensor>() { encodedVectorStateActor, encodedVisualActor }, 1);
            encodedAllCritic = Current.K.concat(new List<Tensor>() { encodedVectorStateCritic, encodedVisualCritic }, 1);
        }



        var criticOutput = new Dense(units: 1, activation: null, use_bias: criticOutputLayerBias, kernel_initializer: new GlorotUniform(scale: criticOutputLayerInitialScale));
        outValue = criticOutput.Call(encodedAllCritic)[0];
        criticWeights.AddRange(criticOutput.weights);

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
