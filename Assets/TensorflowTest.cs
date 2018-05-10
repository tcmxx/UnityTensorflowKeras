using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using static Current;

public class TensorflowTest : MonoBehaviour {

	// Use this for initialization
	void Start () {
        UnityTFTensor t = K.Constant(new float[] { 4, 5, 6},new int[] { 3,1});
        var initializer = new Constant(1);
        var weights = K.Variable(initializer.Call(new int[] { 4, 3 }, DataType.Float));

        var initValue = (float[,])weights.Eval();
        //print(initValue[0,0]);
        t = K.Dot(weights, t);

        var result = (float[,])t.Eval();

        //print(result.GetLength(0));
        //print(result.GetLength(1));
        foreach (var v in result)
        {
            print(v);
        }
        GetComponentInChildren<Text>().text = t.Eval().ToString();
	}
	
	// Update is called once per frame
	void Update () {
		
	}
}
