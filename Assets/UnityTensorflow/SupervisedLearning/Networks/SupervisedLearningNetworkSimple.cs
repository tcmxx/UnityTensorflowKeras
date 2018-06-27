using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

[CreateAssetMenu()]
public class SupervisedLearningNetworkSimple : SupervisedLearningNetwork
{

    public int numHidden = 2;
    public int width = 64;
    public float hiddenWeightsInitialScale = 1;
    public float outputWeightsInitialScale = 0.01f;

    protected List<Tensor> weights;


    public override Tensor BuildNetwork(Tensor inVectorstate, List<Tensor> inVisualState, Tensor inMemery, int outActionSize, SpaceType actionSpace)
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
                var ha = CreateVisualEncoder(v, width, numHidden, "ActorVisualEncoder");
                visualEncodedActor.Add(ha);
            }
            if (inVisualState.Count > 1)
            {
                Debug.LogError("Tensorflow does not have gradient for concat operation in C yet. Please only use one observation.");
                encodedVisualActor = Current.K.concat(visualEncodedActor, 1);
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
            encodedVectorStateActor = CreateContinuousStateEncoder(inVectorstate, width, numHidden, "ActorStateEncoder");
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
        var actorOutput = new Dense(units: outActionSize, activation: null, use_bias: true, kernel_initializer: new GlorotUniform(scale: outputWeightsInitialScale));
        var outAction = actorOutput.Call(encodedAllActor)[0];
        if (actionSpace == SpaceType.discrete)
        {
            outAction = Current.K.softmax(outAction);
        }

        weights.AddRange(actorOutput.weights);

        return outAction;
    }


    protected Tensor CreateVisualEncoder(Tensor visualInput, int hiddenSize, int numLayers, string scope)
    {
        //use the same encoder as in UnityML's python codes
        Tensor temp;
        using (Current.K.name_scope(scope))
        {
            var conv1 = new Conv2D(16, new int[] { 8, 8 }, new int[] { 4, 4 }, activation: new ELU());
            var conv2 = new Conv2D(32, new int[] { 4, 4 }, new int[] { 2, 2 }, activation: new ELU());

            temp = conv1.Call(visualInput)[0];
            temp = conv2.Call(temp)[0];

            var flatten = new Flatten();
            //temp = Current.K.batch_flatten(temp);
            temp = flatten.Call(temp)[0];
            weights.AddRange(conv1.weights);
            weights.AddRange(conv2.weights);
        }

        var hiddenFlat = CreateContinuousStateEncoder(temp, hiddenSize, numLayers, scope);
        return hiddenFlat;
    }

    protected Tensor CreateContinuousStateEncoder(Tensor state, int hiddenSize, int numLayers, string scope)
    {
        var hidden = state;
        using (Current.K.name_scope(scope))
        {
            for (int i = 0; i < numLayers; ++i)
            {
                var layer = new Dense(hiddenSize, new ReLU(), true, kernel_initializer: new GlorotUniform(scale: hiddenWeightsInitialScale));
                hidden = layer.Call(hidden)[0];
                weights.AddRange(layer.weights);
            }
        }

        return hidden;
    }


    public override List<Tensor> GetWeights()
    {
        return weights;
    }
}
