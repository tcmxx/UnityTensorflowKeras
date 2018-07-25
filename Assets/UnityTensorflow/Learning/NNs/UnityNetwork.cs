using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using KerasSharp;
using KerasSharp.Activations;
using System;
using KerasSharp.Engine.Topology;
using KerasSharp.Initializers;
#if UNITY_EDITOR
using UnityEditor;
#endif



public abstract class UnityNetwork : ScriptableObject
{



    [Serializable]
    public class SimpleDenseLayerDef
    {
        public int size = 32;
        public float initialScale = 0.7f;
        public bool useBias = true;
        public Activation.ActivationFunction activationFunction;


        /// <summary>
        /// return the output tensor and list of weights
        /// </summary>
        /// <param name="x">input </param>
        /// <returns>(output tensor, list of weights)</returns>
        public ValueTuple<Tensor, List<Tensor>> Call(Tensor x)
        {
            var layer = new Dense(size, Activation.GetActivationFunction(activationFunction), useBias, kernel_initializer: new GlorotUniform(scale: initialScale));
            var output = layer.Call(x)[0];
            return ValueTuple.Create(output, layer.weights);
        }
    }




    public static ValueTuple<Tensor, List<Tensor>> BuildSequentialLayers(List<SimpleDenseLayerDef> layerDefs, Tensor input)
    {
        List<Tensor> weights = new List<Tensor>();
        Tensor temp = input;
        foreach(var l in layerDefs)
        {
            var result = l.Call(temp);
            temp = result.Item1;
            weights.AddRange(result.Item2);
        }
        return ValueTuple.Create(temp, weights);
    }

    public virtual void OnInspector()
    {
#if UNITY_EDITOR
        Editor.CreateEditor(this).OnInspectorGUI();
#endif
    }
}
