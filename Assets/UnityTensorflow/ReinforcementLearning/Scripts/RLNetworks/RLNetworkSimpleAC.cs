using System.Collections;
using System.Collections.Generic;
using UnityEngine;


[CreateAssetMenu()]
public class RLNetworkSimpleAC : RLNetworkAC
{

    public int actorNNHidden = 1;
    public int actorNNWidth = 128;
    public int criticNNHidden = 1;
    public int criticNNWidth = 128;
    public float weightsInitialScale = 1;


    protected List<Tensor> weights;


    public override void BuildNetwork(Tensor inVectorstate, Tensor inVisualState, Tensor inMemery, Tensor inPrevAction, int outActionSize, 
        out Tensor outAction, out Tensor outValue)
    {

        Debug.Assert(inVisualState == null, "Currently visual state input is not supported by RLNetworkSimpleAC");
        Debug.Assert(inMemery == null, "Currently recurrent input is not supported by RLNetworkSimpleAC");
        Debug.Assert(inPrevAction == null, "Currently previous action input is not supported by RLNetworkSimpleAC");

        var temp = inVectorstate;


        weights = new List<Tensor>();
        //create the actor layers
        for (int i = 0; i < actorNNHidden; ++i)
        {
            var layer = new Dense(actorNNWidth, new ReLU(), true, kernel_initializer: new GlorotUniform(scale: weightsInitialScale));
            temp = layer.Call(temp)[0];
            weights.AddRange(layer.weights);
        }

        var actorOutput = new Dense(units: outActionSize, activation: null, use_bias: true, kernel_initializer: new GlorotUniform(scale: weightsInitialScale));
        outAction = actorOutput.Call(temp)[0];
        weights.AddRange(actorOutput.weights);

        temp = inVectorstate;
        //create the critic layers
        for (int i = 0; i < criticNNHidden; ++i)
        {
            var layer = new Dense(criticNNWidth, new ReLU(), true, kernel_initializer: new GlorotUniform(scale: weightsInitialScale));
            temp = layer.Call(temp)[0];
            weights.AddRange(layer.weights);
        }

        var criticOutput = new Dense(units: 1, activation: null, use_bias: true, kernel_initializer: new GlorotUniform(scale: weightsInitialScale));
        outValue = criticOutput.Call(temp)[0];
        weights.AddRange(criticOutput.weights);
    }




    public override List<Tensor> GetWeights()
    {
        return weights;
    }
}
