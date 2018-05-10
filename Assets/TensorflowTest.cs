using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using static Current;

public class TensorflowTest : MonoBehaviour {

	// Use this for initialization
	void Start () {

        //create model first
        var input= K.Placeholder(new int?[] { -1 ,3 });
        var target = K.Placeholder(new int?[] { -1,1});
        var weight = K.Variable((new Constant(1)).Call(new int[] { 3, 1 }, DataType.Float));
        var output = K.Dot(input, weight);
        output = K.Reshape(output, new int[] { -1});
        target = K.Reshape(target, new int[] { -1 });

        var lossM = new MeanSquareError();
        var loss = lossM.Call(target, output);

        loss = K.Constant(1.0f) * loss;

        //training related
        var weights = new List<UnityTFTensor>();
        weights.Add(weight);
        var optimizer = new SGD();
        //var updates = optimizer.get_updates(weights, new Dictionary<UnityTFTensor, IWeightConstraint>(),loss);

        var inputs = new List<UnityTFTensor>();
        inputs.Add(input);
        inputs.Add(target);
        var outputs = new List<UnityTFTensor>();
        outputs.Add(loss);

        
        var function = K.Function(inputs, outputs, null, "Train");

        var inputData = new List<Array>();
        inputData.Add(new float[] { 1.2f, 3.3f, 4.3f,5,5,5 });
        inputData.Add(new float[] { 2,10 });
        var functionResult = function.Call(inputData);
        
        var test = K.Placeholder(new int?[] { -1 });
        test = K.Constant(1.0f) * test;

        K.ExportGraphDef("SavedGraph/test.pb");

        float resultLoss = (float)functionResult[0].Eval();
        print(resultLoss);
    }
	
	// Update is called once per frame
	void Update () {
		
	}
}
