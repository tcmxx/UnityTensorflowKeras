using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using static Current;

public class TensorflowTest : MonoBehaviour {

	// Use this for initialization
	void Start () {

        //TestBasicBackendAndOptimizerAndExportGraph();
        //TestLayer();

        TestConv2D();
    }
	


	// Update is called once per frame
	void Update () {
		
	}


    public void TestBasicBackendAndOptimizerAndExportGraph()
    {
        //create model first
        var input = K.Placeholder(new int?[] { -1, 3 });
        var target = K.Placeholder(new int?[] { -1, 1 });
        var weight = K.Variable((new Constant(1)).Call(new int[] { 3, 1 }, DataType.Float));
        var output = K.Dot(input, weight);
        output = K.Reshape(output, new int[] { -1 });
        target = K.Reshape(target, new int[] { -1 });

        var lossM = new MeanSquareError();
        var loss = lossM.Call(target, output);

        loss = K.Constant(1.0f) * loss;

        //training related
        var weights = new List<UnityTFTensor>();
        weights.Add(weight);
        var optimizer = new SGD();
        var updates = optimizer.get_updates(weights, new Dictionary<UnityTFTensor, IWeightConstraint>(), loss);

        var inputs = new List<UnityTFTensor>();
        inputs.Add(input);
        inputs.Add(target);
        var outputs = new List<UnityTFTensor>();
        outputs.Add(loss);


        var function = K.Function(inputs, outputs, updates, "Train");

        var inputData = new List<Array>();
        inputData.Add(new float[] { 1.2f, 3.3f, 4.3f, 5, 5, 5 });
        inputData.Add(new float[] { 2, 10 });

        for (int i = 0; i < 20; ++i)
        {
            var functionResult = function.Call(inputData);
            float resultLoss = (float)functionResult[0].Eval();
            print(resultLoss);
        }
        K.ExportGraphDef("SavedGraph/test.pb");
    }

    public void TestLayer()
    {
        var inputLayer = UnityTFUtils.Input(shape:new int?[] { 3 });

        var dense1 = new Dense(10, new ReLU(), true);
        var dense2 = new Dense(1, new ReLU(), true);

        var target = UnityTFUtils.Input(shape: new int?[] { 1 });
        
        var o = dense1.Call(inputLayer[0]);
        o = dense2.Call(o[0]);

        var lossM = new MeanSquareError();

        var loss = lossM.Call(target[0], o[0]);



        

        K.ExportGraphDef("SavedGraph/testLayer.pb");
    }


    public void TestConv2D()
    {
        var inputLayer = UnityTFUtils.Input(shape: new int?[] { 32,32,3 });

        var conv1 = new Conv2D(16, new int[] { 3, 3 }, padding: PaddingType.Same, activation:new ReLU());
        var conv2 = new Conv2D(3, new int[] { 3, 3 }, padding: PaddingType.Same, activation: new ReLU());

        var target = UnityTFUtils.Input(shape: new int?[] { 32, 32, 3 });


        var pred = conv2.Call(conv1.Call(inputLayer[0])[0])[0];
        var lossM = new MeanSquareError();

        var loss = lossM.Call(target[0], pred);


        K.ExportGraphDef("SavedGraph/convLayer.pb");
    }
}
