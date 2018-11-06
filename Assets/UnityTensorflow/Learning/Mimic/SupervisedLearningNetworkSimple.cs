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

[CreateAssetMenu(menuName = "ML-Agents/InternalLearning/sl/SupervisedLearningNetworkSimple")]
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

    public override ValueTuple<Tensor, Tensor> BuildNetworkForContinuousActionSapce(Tensor inVectorObservation, List<Tensor> inVisualObservation, Tensor inMemery, int outActionSize)
    {
        var encodedActor = CreateCommonLayers(inVectorObservation, inVisualObservation, inMemery, null);


        //outputs
        var actorOutput = new Dense(units: outActionSize, activation: null, use_bias: outputLayerBias, kernel_initializer: new GlorotUniform(scale: outputLayerInitialScale));
        var outAction = actorOutput.Call(encodedActor)[0];

        weights.AddRange(actorOutput.weights);

        Tensor outVar = null;
        if (useVarianceForContinuousAction)
        {
            var logSigmaSq = new Dense(units: 1, activation: null, use_bias: outputLayerBias, kernel_initializer: new GlorotUniform(scale: outputLayerInitialScale));
            outVar = Current.K.exp(logSigmaSq.Call(encodedActor)[0]) + minStd * minStd;
            weights.AddRange(logSigmaSq.weights);
        }

        return ValueTuple.Create(outAction, outVar);

    }

    public override List<Tensor> BuildNetworkForDiscreteActionSpace(Tensor inVectorObs, List<Tensor> inVisualObs, Tensor inMemery, int[] outActionSizes)
    {
        Tensor encodedAllActor = CreateCommonLayers(inVectorObs, inVisualObs, inMemery, null);

        List<Tensor> policy_branches = new List<Tensor>();
        foreach (var size in outActionSizes)
        {
            var tempOutput = new Dense(units: size, activation: null, use_bias: outputLayerBias, kernel_initializer: new VarianceScaling(scale: outputLayerInitialScale));
            policy_branches.Add(tempOutput.Call(encodedAllActor)[0]);
            weights.AddRange(tempOutput.weights);
        }
        return policy_branches;
    }

    protected Tensor CreateCommonLayers(Tensor inVectorObs, List<Tensor> inVisualObs, Tensor inMemery, Tensor inPrevAction)
    {

        weights = new List<Tensor>();

        ValueTuple<Tensor, List<Tensor>> actorEncoded;
        actorEncoded = CreateObservationStream(inVectorObs, hiddenLayers, inVisualObs, inMemery, inPrevAction, "Actor");

        weights.AddRange(actorEncoded.Item2);

        return actorEncoded.Item1;
    }


    protected ValueTuple<Tensor, List<Tensor>> CreateObservationStream(Tensor inVectorObs, List<SimpleDenseLayerDef> layerDefs, List<Tensor> inVisualObs, Tensor inMemery, Tensor inPrevAction, string encoderName)
    {
        Debug.Assert(inMemery == null, "Currently recurrent input is not supported by RLNetworkSimpleAC");
        Debug.Assert(inPrevAction == null, "Currently previous action input is not supported by RLNetworkSimpleAC");
        Debug.Assert(!(inVectorObs == null && inVisualObs == null), "Network need at least one vector observation or visual observation");

        List<Tensor> allWeights = new List<Tensor>();
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
        return weights;
    }


}
