using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using KerasSharp;
using KerasSharp.Activations;
using System;
using KerasSharp.Engine.Topology;
using KerasSharp.Initializers;
using KerasSharp.Backends;

public abstract class UnityNetwork : ScriptableObject
{



    [Serializable]
    public class SimpleDenseLayerDef
    {
        public int size = 32;
        public float initialScale = 1f;
        public bool useBias = true;
        public Activation.ActivationFunction activationFunction;
        public SimpleDenseLayerDef()
        {
            size = 32;
        }

        public void Reinitialize()
        {
            size = 32;
            initialScale = 1;
            useBias = true;
            activationFunction = Activation.ActivationFunction.Swish;
        }

        /// <summary>
        /// return the output tensor and list of weights
        /// </summary>
        /// <param name="x">input </param>
        /// <returns>(output tensor, list of weights)</returns>
        public ValueTuple<Tensor, List<Tensor>> Call(Tensor x)
        {
            var layer = new Dense(size, Activation.GetActivationFunction(activationFunction), useBias, kernel_initializer: new VarianceScaling(scale: initialScale));
            var output = layer.Call(x)[0];
            return ValueTuple.Create(output, layer.weights);
        }
    }



    /// <summary>
    /// Build neural network based on the lsit of layer defs.
    /// </summary>
    /// <param name="layerDefs">layer definitions</param>
    /// <param name="input">input tensor</param>
    /// <param name="scope">name scope</param>
    /// <returns>value tuple of (output tensor, list of weights)</returns>
    public static ValueTuple<Tensor, List<Tensor>> BuildSequentialLayers(List<SimpleDenseLayerDef> layerDefs, Tensor input, string scope = null)
    {
        NameScope nameScppe = null;
        if(!string.IsNullOrEmpty(scope))
            nameScppe = Current.K.name_scope(scope);

        List<Tensor> weights = new List<Tensor>();
        Tensor temp = input;
        foreach(var l in layerDefs)
        {
            var result = l.Call(temp);
            temp = result.Item1;
            weights.AddRange(result.Item2);
        }

        if (nameScppe != null)
            nameScppe.Dispose();
        return ValueTuple.Create(temp, weights);
    }

}
