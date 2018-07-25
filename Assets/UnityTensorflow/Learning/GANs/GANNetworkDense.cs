using System.Collections;
using System.Collections.Generic;
using KerasSharp;
using KerasSharp.Models;
using KerasSharp.Engine.Topology;
using KerasSharp.Initializers;
using UnityEngine;

using static KerasSharp.Backends.Current;
using KerasSharp.Activations;

[CreateAssetMenu()]
public class GANNetworkDense : GANNetwork
{

    public List<SimpleDenseLayerDef> generatorHiddenLayers;
    public List<SimpleDenseLayerDef> discriminatorHiddenLayers;

    public float generatorOutputLayerInitialScale = 0.1f;
    public bool generatorOutputLayerBias = true;

    public float discriminatorOutputLayerInitialScale = 0.1f;
    public bool discriminatorOutputLayerBias = true;

    protected List<Tensor> generatorWeights;
    protected List<Tensor> discriminatorWeights;


    public override void BuildNetwork(Tensor inputCondition, Tensor inputNoise, Tensor inputTargetToJudge, int[] outputShape, out Tensor generatorOutput, out Tensor discriminatorOutputExternal, out Tensor discriminatorOutputFromGenerator)
    {
        Debug.Assert(outputShape.Length == 1, "outputShape need to have 1 d only for dense gan");
        generatorOutput = CreateGenerator(inputCondition, inputNoise, outputShape[0]);

        CreateDiscriminators(inputTargetToJudge, generatorOutput, inputCondition, out discriminatorOutputExternal, out discriminatorOutputFromGenerator);
    }

    public override List<Tensor> GetDiscriminatorWeights()
    {
        return discriminatorWeights;
    }

    public override List<Tensor> GetGeneratorWeights()
    {
        return generatorWeights;
    }


    protected void CreateDiscriminators(Tensor inputExternal,  Tensor inputFromGenerator, Tensor inputCondition,out Tensor discriminatorOutputExternal, out Tensor discriminatorOutputFromGenerator)
    {
        Tensor inputAllReal = null;
        List<Tensor> inputListExternal = null;
        List<Tensor> inputListFromGenerator = null;
        if (inputCondition != null)
        {
            inputListExternal = new List<Tensor>() { inputCondition, inputExternal };
            inputListFromGenerator = new List<Tensor>() { inputCondition, inputFromGenerator };
            inputAllReal = new Concat(1).Call(inputListExternal)[0];
        }
        else
        {
            inputListFromGenerator = new List<Tensor>() { inputFromGenerator };
            inputListExternal = new List<Tensor>() {inputExternal };
            inputAllReal = inputExternal;
        }

        var beforeOutput = BuildSequentialLayers(discriminatorHiddenLayers, inputAllReal);

        var outputLayer = new Dense(1, new Sigmoid(), discriminatorOutputLayerBias, kernel_initializer: new GlorotUniform(scale: discriminatorOutputLayerInitialScale));
        discriminatorOutputExternal = outputLayer.Call(beforeOutput.Item1)[0];

        Model model = new Model(inputListExternal, new List<Tensor>() { discriminatorOutputExternal });

        discriminatorOutputFromGenerator = model.Call(inputListFromGenerator)[0];

        discriminatorWeights = model.weights;

    }

    protected Tensor CreateGenerator(Tensor inputCondition, Tensor inputNoise, int outputSize)
    {
        Debug.Assert(inputCondition != null || inputNoise != null, "GAN needs at least one of input condition or input noise ");

        Tensor inputAll = null;
        List<Tensor> inputList = null;
        if(inputNoise != null && inputCondition != null)
        {
            inputList = new List<Tensor>() { inputCondition, inputNoise };
            inputAll = new Concat(1).Call(inputList)[0];
        }
        else if(inputNoise != null)
        {
            inputAll = inputNoise;
            inputList = new List<Tensor>() { inputNoise };
        }
        else
        {
            inputAll = inputCondition;
            inputList = new List<Tensor>() { inputCondition };
        }

        var beforeOutput = BuildSequentialLayers(generatorHiddenLayers, inputAll);

        var outputLayer = new Dense(outputSize, null, generatorOutputLayerBias, kernel_initializer: new GlorotUniform(scale: generatorOutputLayerInitialScale));
        var output = outputLayer.Call(beforeOutput.Item1)[0];

        generatorWeights = new List<Tensor>();
        generatorWeights.AddRange(beforeOutput.Item2);
        generatorWeights.AddRange(outputLayer.weights);

        return output;
        
    }
}
