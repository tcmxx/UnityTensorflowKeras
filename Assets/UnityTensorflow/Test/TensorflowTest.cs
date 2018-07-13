using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Linq;
using Accord.Math;

using static KerasSharp.Backends.Current;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;
using KerasSharp.Initializers;
using KerasSharp.Losses;
using KerasSharp.Engine.Topology;
using KerasSharp.Optimizers;
using KerasSharp;
using KerasSharp.Constraints;
using KerasSharp.Activations;
using KerasSharp.Models;
using KerasSharp.Backends;

public class TensorflowTest : MonoBehaviour {

	// Use this for initialization
	void Start () {
        

        //TestBasicBackendAndOptimizerAndExportGraph();
        //TestLayer();

        //TestConv2D();

        //
        //TestSetAndGetValue();

        //TestModelCompileAndFit();

        //TestDataBuffer();

        //print(Path.GetFullPath("Set/setset/set.ser"));

        TestConcatGradient();
    }
	


	// Update is called once per frame
	void Update () {
		
	}



    public void TestDataBuffer()
    {
        DataBuffer testBuffer = new DataBuffer(100, new DataBuffer.DataInfo("Test1", typeof(float), new int[] { 1, 2 }),
            new DataBuffer.DataInfo("Test2", typeof(float), new int[] { 4 }));

        List<ValueTuple<string, Array>> dataToAdd = new List<ValueTuple<string, Array>>();
        dataToAdd.Add(ValueTuple.Create<string, Array>("Test1", new float[2, 1, 2] { { { 1, 2 } },{ { 3, 4 } } }));
        dataToAdd.Add(ValueTuple.Create<string, Array>("Test2", new float[8] { 1,2,3,4,5,6,7,8 }));

        testBuffer.AddData(dataToAdd.ToArray());

        //var result = testBuffer.FetchDataAt(1, ValueTuple.Create("Test1", 2, "Test1"), ValueTuple.Create("Test2", 2, "Test2"));



        var formatter = new BinaryFormatter();
        using (var stream = new FileStream("test.test", FileMode.Create, FileAccess.Write, FileShare.None))
            formatter.Serialize(stream, testBuffer);

        DataBuffer t2;
        using (var stream = new FileStream("test.test", FileMode.Open, FileAccess.Read, FileShare.Read))
            t2 = (DataBuffer)formatter.Deserialize(stream);

        Debug.Assert(testBuffer.CurrentCount == t2.CurrentCount, "Wrong serialization");
        Debug.Assert(testBuffer.MaxCount == t2.MaxCount, "Wrong serialization");

        var t2Result = t2.FetchDataAt(1,  ValueTuple.Create("Test2", 2, "Test2"));
        var t1Result = t2.FetchDataAt(1, ValueTuple.Create("Test2", 2, "Test2"));

        bool resultEquals = t1Result["Test2"].Equals(t2Result["Test2"]);
        Debug.Assert(resultEquals, "Data not matching.Wrong serialization.");



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

        for (int i = 0; i < 10; ++i)
        {
            var functionResult = function.Call(inputData);
            float resultLoss = (float)functionResult[0].eval();
            print(resultLoss);
            var opWeights = optimizer.get_weights();
            foreach(var w in opWeights)
            {
                string toPrint = "";
                for(int j = 0; j < w.Length; ++j)
                {
                    if(w is float[])
                        toPrint += " " + w.GetValue(j);
                    else
                        toPrint += " " + w.GetValue(j,0);
                }
                print("Weight:"+ toPrint);
            }
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
            ValueTuple.Create(weight1,(Array)inputValue1),ValueTuple.Create(weight2,(Array)inputValue2),
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


        
        // Evaluate the model
        double[] scores = model.evaluate(x, y);
        Debug.Log("Eval results: " + string.Join(",", scores));
        //scores = model.evaluate(x, y); scores = model.evaluate(x, y); 
        //Debug.Log($"{model.metrics_names[1]}: {scores[1] * 100}");

        ((UnityTFBackend)K).ExportGraphDef("SavedGraph/sequentialtest.pb");
    }

    //test the concat gradient of the backend.
    //This is tested because the Tensorflow c++ concat gradient is not officially impelmented. I wrote my own version and built it. Might be wrong. 
    public bool TestConcatGradient()
    {
        //create model first
        var input1 = K.placeholder(new int?[] { -1, 1 });
        var input2 = K.placeholder(new int?[] { -1, 1 });
        var input3 = K.placeholder(new int?[] { -1, 1 });
        var weight1 = K.variable((new Constant(1)).Call(new int[] { 1, 1 }, DataType.Float));
        var weight2 = K.variable((new Constant(1)).Call(new int[] { 1, 2 }, DataType.Float));
        var weight3 = K.variable((new Constant(1)).Call(new int[] { 1, 3 }, DataType.Float));

        var output1 = K.dot(input1, weight1);
        var output2 = K.dot(input2, weight2);
        var output3 = K.dot(input3, weight3);

        var concated = K.concat(new List<Tensor>() { output1, output2, output3 }, 1);

        var input4 = K.placeholder(new int?[] { -1, 6 });

        var output = K.mul(concated, input4);

        var gradients = K.gradients(output, new List<Tensor>() { weight1, weight2, weight3 });
        var g1 = K.identity(gradients[0], "finalGradient1");
        var g2 = K.identity(gradients[1], "finalGradient2");
        var g3 = K.identity(gradients[2], "finalGradient3");

        var inputs = new List<Tensor>();
        inputs.Add(input1); inputs.Add(input2); inputs.Add(input3); inputs.Add(input4);
        var outputs = new List<Tensor>();
        outputs.Add(g1); outputs.Add(g2); outputs.Add(g3);

        var function = K.function(inputs, outputs, null, "GetGradients");

        var inputData = new List<Array>();
        inputData.Add(new float[] { 1f});
        inputData.Add(new float[] { 2f});
        inputData.Add(new float[] { 3f});
        inputData.Add(new float[] { 1f,2f,3f,4f,5f,6f });

        var functionResult = function.Call(inputData);
        float[,] resultg1 = (float[,])functionResult[0].eval();
        float[,] resultg2 = (float[,])functionResult[1].eval();
        float[,] resultg3 = (float[,])functionResult[2].eval();

        bool pass = resultg1[0, 0] == 1 
            && resultg2[0, 0] == 4 && resultg2[0, 1] == 6
            && resultg3[0, 0] == 12 && resultg3[0, 1] == 15 && resultg3[0, 2] == 18;

        Debug.Assert(pass, "TestConcatGradient: Wrong gradient, test failed!");
        ((UnityTFBackend)K).ExportGraphDef("SavedGraph/TestConcatGradient.pb");
        return pass;
    }
}
