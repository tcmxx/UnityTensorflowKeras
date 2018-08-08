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
public class SupervisedLearningNetworkSimple : SupervisedLearningNetwork
{

    public List<SimpleDenseLayerDef> hiddenLayers;


    public float outputLayerInitialScale = 0.1f;
    public bool outputLayerBias = true;
    public float visualEncoderInitialScale = 0.1f;
    public bool visualEncoderBias = true;


    //public int numHidden = 2;
    //public int width = 64;
    //public float hiddenWeightsInitialScale = 1;
    //public float outputWeightsInitialScale = 0.01f;
    public bool useVarianceForContinuousAction = false;
    public float minStd = 0.1f;

    protected List<Tensor> weights;


    public override ValueTuple<Tensor, Tensor> BuildNetwork(Tensor inVectorstate, List<Tensor> inVisualState, Tensor inMemery, int outActionSize, SpaceType actionSpace)
    {

        Debug.Assert(inMemery == null, "Currently recurrent input is not supported by SupervisedLearningNetworkSimple");
        Debug.Assert(!(inVectorstate == null && inVisualState == null), "Network need at least one vector observation or visual observation");


        weights = new List<Tensor>();

        //visual encoders
        Tensor encodedVisualActor = null;
        if (inVisualState != null)
        {
            List<Tensor> visualEncodedActor = new List<Tensor>();
            foreach (var v in inVisualState)
            {
                var ha = CreateVisualEncoder(v, hiddenLayers, "ActorVisualEncoder");
                visualEncodedActor.Add(ha);
            }
            if (inVisualState.Count > 1)
            {
                //Debug.LogError("Tensorflow does not have gradient for concat operation in C yet. Please only use one observation.");
                encodedVisualActor = Current.K.stack(visualEncodedActor, 1);
                encodedVisualActor = Current.K.batch_flatten(encodedVisualActor);
            }
            else
            {
                encodedVisualActor = visualEncodedActor[0];
            }


        }

        //vector states encode
        Tensor encodedVectorStateActor = null;
        if (inVectorstate != null)
        {
            var hiddens = BuildSequentialLayers(hiddenLayers, inVectorstate, "ActorStateEncoder");
            encodedVectorStateActor = hiddens.Item1;
            weights.AddRange(hiddens.Item2);
        }

        //concat all inputs
        Tensor encodedAllActor = null;

        if (inVisualState == null && inVectorstate != null)
        {
            encodedAllActor = encodedVectorStateActor;
        }
        else if (inVisualState != null && inVectorstate == null)
        {
            encodedAllActor = encodedVisualActor;
        }
        else if (inVisualState != null && inVectorstate != null)
        {
            //Debug.LogError("Tensorflow does not have gradient for concat operation in C yet. Please only use one observation.");
            encodedAllActor = Current.K.concat(new List<Tensor>() { encodedVectorStateActor, encodedVisualActor }, 1);
        }


        //outputs
        var actorOutput = new Dense(units: outActionSize, activation: null, use_bias: outputLayerBias, kernel_initializer: new GlorotUniform(scale: outputLayerInitialScale));
        var outAction = actorOutput.Call(encodedAllActor)[0];
        if (actionSpace == SpaceType.discrete)
        {
            outAction = Current.K.softmax(outAction);
        }

        weights.AddRange(actorOutput.weights);

        Tensor outVar = null;
        if(useVarianceForContinuousAction && actionSpace == SpaceType.continuous)
        {
            var logSigmaSq = new Dense(units: 1, activation: null, use_bias: outputLayerBias, kernel_initializer: new GlorotUniform(scale: outputLayerInitialScale));
            outVar = Current.K.exp(logSigmaSq.Call(encodedAllActor)[0]) +minStd*minStd;
            weights.AddRange(logSigmaSq.weights);
        }

        return ValueTuple.Create(outAction,outVar);
    }


    protected Tensor CreateVisualEncoder(Tensor visualInput, List<SimpleDenseLayerDef> denseLayers, string scope)
    {
        //use the same encoder as in UnityML's python codes
        Tensor temp;
        using (Current.K.name_scope(scope))
        {
            var conv1 = new Conv2D(16, new int[] { 8, 8 }, new int[] { 4, 4 }, use_bias: visualEncoderBias, kernel_initializer: new GlorotUniform(scale: visualEncoderInitialScale), activation: new ELU());
            var conv2 = new Conv2D(32, new int[] { 4, 4 }, new int[] { 2, 2 }, use_bias: visualEncoderBias, kernel_initializer: new GlorotUniform(scale: visualEncoderInitialScale), activation: new ELU());

            temp = conv1.Call(visualInput)[0];
            temp = conv2.Call(temp)[0];

            var flatten = new Flatten();
            //temp = Current.K.batch_flatten(temp);
            temp = flatten.Call(temp)[0];
            weights.AddRange(conv1.weights);
            weights.AddRange(conv2.weights);
        }

        //var hiddenFlat = CreateContinuousStateEncoder(temp, hiddenSize, numLayers, scope);
        var output = BuildSequentialLayers(denseLayers, temp, scope);
        var hiddenFlat = output.Item1;
        weights.AddRange(output.Item2);
        return hiddenFlat;
    }



    public override List<Tensor> GetWeights()
    {
        return weights;
    }
}
