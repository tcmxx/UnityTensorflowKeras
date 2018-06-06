using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Linq;
using Accord;

using static Current;

public class TensorflowTest : MonoBehaviour {

	// Use this for initialization
	void Start () {

        //TestBasicBackendAndOptimizerAndExportGraph();
        //TestLayer();

        //TestConv2D();

        //
        //TestSetAndGetValue();

        TestModelCompileAndFit();
    }
	


	// Update is called once per frame
	void Update () {
		
	}


    public void TestBasicBackendAndOptimizerAndExportGraph()
    {
        //create model first
        var input = K.placeholder(new int?[] { -1, 3 });
        var target = K.placeholder(new int?[] { -1, 1 });
        var weight = K.variable((new Constant(1)).Call(new int[] { 3, 1 }, DataType.Float));
        var output = K.dot(input, weight);
        output = K.reshape(output, new int[] { -1 });
        target = K.reshape(target, new int[] { -1 });

        var lossM = new MeanSquareError();
        var loss = lossM.Call(target, output);

        loss = K.constant(1.0f) * loss;

        //training related
        var weights = new List<Tensor>();
        weights.Add(weight);
        var optimizer = new SGD();
        var updates = optimizer.get_updates(weights, new Dictionary<Tensor, IWeightConstraint>(), loss);

        var inputs = new List<Tensor>();
        inputs.Add(input);
        inputs.Add(target);
        var outputs = new List<Tensor>();
        outputs.Add(loss);


        var function = K.function(inputs, outputs, updates, "Train");

        var inputData = new List<Array>();
        inputData.Add(new float[] { 1.2f, 3.3f, 4.3f, 5, 5, 5 });
        inputData.Add(new float[] { 2, 10 });

        for (int i = 0; i < 20; ++i)
        {
            var functionResult = function.Call(inputData);
            float resultLoss = (float)functionResult[0].eval();
            print(resultLoss);
        }
        ((UnityTFBackend)K).ExportGraphDef("SavedGraph/test.pb");
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

        lossM.Call(target[0], o[0]);





        ((UnityTFBackend)K).ExportGraphDef("SavedGraph/testLayer.pb");
    }


    public void TestConv2D()
    {
        var inputLayer = UnityTFUtils.Input(shape: new int?[] { 32,32,3 });

        var conv1 = new Conv2D(16, new int[] { 3, 3 }, padding: PaddingType.Same, activation:new ReLU());
        var conv2 = new Conv2D(3, new int[] { 3, 3 }, padding: PaddingType.Same, activation: new ReLU());

        var target = UnityTFUtils.Input(shape: new int?[] { 32, 32, 3 });


        var pred = conv2.Call(conv1.Call(inputLayer[0])[0])[0];
        var lossM = new MeanSquareError();

        lossM.Call(target[0], pred);


        ((UnityTFBackend)K).ExportGraphDef("SavedGraph/convLayer.pb");
    }

    public void TestSetAndGetValue()
    {
        var weight1 = K.variable((new Constant(1)).Call(new int[] { 3 }, DataType.Float));
        var weight2 = K.variable((new Constant(1)).Call(new int[] { 5 }, DataType.Float));
        //call eval once for intiaization
        K.batch_get_value(new List<Tensor>() { weight1, weight2 });

        print("Test SetValue() and GetValue()");
        var inputValue1 = new float[] { 8.8f, 9.9f, 1.1f };
        K.set_value(weight1, inputValue1);

        var result1 = (float[])K.get_value(weight1);

        print("---Input value:" + string.Join(", ", inputValue1));
        print("---Output value:" + string.Join(", ", result1));

        print("Test "+ (result1.SequenceEqual(inputValue1)?"Passed":"Failed"));



        print("Test batch_set_value() and batch_get_value()");
        inputValue1 = new float[] { 2.2f, 3.9f, 4.1f };
        var inputValue2 = new float[] { 4.2f, 3.2f, 14.5f, 44.5f, 74.3f };
        K.batch_set_value(new List<ValueTuple<Tensor, Array>>() {
            ValueTuple.Create(weight1,inputValue1),ValueTuple.Create(weight2,inputValue2),
        });

        var resultBatch = K.batch_get_value(new List<Tensor>() { weight1 , weight2 });

        
        print("---Input value1:" + string.Join(", ", inputValue1));
        print("---Output value1:" + string.Join(", ", (float[])resultBatch[0]));
        print("---Input value2:" + string.Join(", ", inputValue2));
        print("---Output value2:" + string.Join(", ", (float[])resultBatch[1]));

        print("Test " + ((inputValue1.SequenceEqual((float[])resultBatch[0])
             && inputValue2.SequenceEqual((float[])resultBatch[1])) ? "Passed" : "Failed"));

    }


    public void TestModelCompileAndFit()
    {

        print("Test base Squential Model");

        float[,] x = { { 2, 3, 4, 5 },{ 4,3,2,1}, { 8, 44, 22, 11 }, { 1, 3, 3, 1 } };
        float[,] y = { { 0.2f,0.2f }, { 0.4f, 0.4f }, { 0.8f, 0.8f }, { 0.1f, 0.1f } };
        var model = new Sequential();
        model.Add(new Dense(12, input_dim: 4, activation: new ReLU()));
        model.Add(new Dense(8, activation: new ReLU()));
        model.Add(new Dense(2,  activation: new Sigmoid()));

        // Compile the model (for the moment, only the mean square 
        // error loss is supported, but this should be solved soon)
        model.Compile(loss: new MeanSquareError(),
            optimizer: new Adam(0.001));

        print("fit 1");
        model.fit(x, y, batch_size: 4,epochs:30, verbose:1);

        print("fit 2");
        model.fit(x, y, batch_size: 4, epochs: 30, verbose: 1);
        // Use the model to make predictions
        //var test = model.predict(x)[0];
        //float[,] pred = model.predict(x)[0].To<float[,]>();


        ((UnityTFBackend)K).ExportGraphDef("SavedGraph/sequentialtest.pb");
        // Evaluate the model
        double[] scores = model.evaluate(x, y);
        //Debug.Log($"{model.metrics_names[1]}: {scores[1] * 100}");
    }
}
