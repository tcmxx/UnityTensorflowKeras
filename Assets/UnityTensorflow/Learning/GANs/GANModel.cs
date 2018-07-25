
using KerasSharp;
using KerasSharp.Backends;
using KerasSharp.Engine.Topology;
using KerasSharp.Losses;
using KerasSharp.Models;
using KerasSharp.Optimizers;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

using static KerasSharp.Backends.Current;
using System;

public class GANModel : MonoBehaviour {

    public GANNetwork network;

    protected Tensor inputCorrectLabel;

    public float generatorL2LossWeight = 1;

    protected Function trainGeneratorFunction;
    protected Function trainDiscriminatorFunction;
    protected Function generateFunction;

    public bool HasNoiseInput { get; private set; }
    public bool HasConditionInput { get; private set; }
    public bool HasGeneratorL2Loss { get; private set; }
    private void Start()
    {
        Initialize(new int[] { 2 }, new int[] { 1 }, null, true);

        ((UnityTFBackend)K).ExportGraphDef("SavedGraph/test.pb");
    }

    public void Initialize(int[] outputShape, int[] inputNoiseShape = null, int[] inputConditionShape = null, bool hasGeneratorL2Loss = false)
    {
        HasNoiseInput = inputNoiseShape != null;
        HasConditionInput = inputConditionShape != null;
        HasGeneratorL2Loss = hasGeneratorL2Loss;

        Tensor inputCondition = null;
        if(inputConditionShape != null)
            inputCondition = UnityTFUtils.Input(inputConditionShape.Select((t)=>(int?)t).ToArray(), name:"InputConditoin")[0];
        Tensor inputNoise = null;
        if (inputNoiseShape != null)
            inputNoise = UnityTFUtils.Input(inputNoiseShape.Select((t) => (int?)t).ToArray(), name: "InputNoise")[0];

        Debug.Assert(inputNoiseShape != null || inputConditionShape != null, "GAN needs at least one of noise or condition input");

        Tensor inputTargetToJudge = UnityTFUtils.Input(outputShape.Select((t) => (int?)t).ToArray(), name: "InputTargetToJudge")[0];
        



        Tensor generatorOutput, disOutForGenerator, dicOutTarget;

        network.BuildNetwork(inputCondition, inputNoise, inputTargetToJudge, outputShape, out generatorOutput, out dicOutTarget, out disOutForGenerator);



        //build the loss
        //generator gan loss
        Tensor genGANLoss = K.constant(0.0f, new int[] { },DataType.Float) - K.mean(K.binary_crossentropy(disOutForGenerator, K.constant(0.0f, new int[] { }, DataType.Float), true));
        Tensor genLoss = genGANLoss;
        //generator l2Loss if use it
        Tensor l2Loss = null;
        Tensor inputGeneratorTarget = null;
        Tensor inputL2LossWeight = null;
        if (hasGeneratorL2Loss)
        {
            inputL2LossWeight = UnityTFUtils.Input(batch_shape: new int?[] { }, name: "l2LossWeight", dtype: DataType.Float)[0];
            inputGeneratorTarget = UnityTFUtils.Input(outputShape.Select((t) => (int?)t).ToArray(), name: "GeneratorTarget")[0];
            l2Loss = K.mul(inputL2LossWeight, K.mean(new MeanSquareError().Call(inputGeneratorTarget, generatorOutput)));
            genLoss = genGANLoss + l2Loss;
        }

        //discriminator loss
        inputCorrectLabel = UnityTFUtils.Input(new int?[] { 1 },name:"InputCorrectLabel")[0];
        Tensor discLoss = K.mean(K.binary_crossentropy(dicOutTarget, inputCorrectLabel, true));



        //create the Functions inputs
        List<Tensor> generatorTrainInputs = new List<Tensor>();
        List<Tensor> generateInputs = new List<Tensor>();
        List<Tensor> discriminatorTrainInputs = new List<Tensor>();
        discriminatorTrainInputs.Add(inputTargetToJudge);
        discriminatorTrainInputs.Add(inputCorrectLabel);
        if (inputCondition != null)
        {
            generatorTrainInputs.Add(inputCondition);
            generateInputs.Add(inputCondition);
            discriminatorTrainInputs.Add(inputCondition);
        }
        if(inputNoise != null)
        {
            generatorTrainInputs.Add(inputNoise);
            generateInputs.Add(inputNoise);
        }
        if (hasGeneratorL2Loss)
        {
            generatorTrainInputs.Add(inputGeneratorTarget);
            generatorTrainInputs.Add(inputL2LossWeight);
        }

        var genOptimizer = new Adam(lr: 0.001);
        var generatorUpdate =  genOptimizer.get_updates(network.GetGeneratorWeights(), null, genLoss); ;
        trainGeneratorFunction = K.function(generatorTrainInputs, new List<Tensor> { genLoss }, generatorUpdate, "GeneratorUpdateFunction");

        var discOptimizer = new Adam(lr: 0.001);
        var discriminatorUpdate = discOptimizer.get_updates(network.GetDiscriminatorWeights(), null, discLoss); ;
        trainDiscriminatorFunction = K.function(discriminatorTrainInputs, new List<Tensor> { discLoss }, discriminatorUpdate, "DiscriminatorUpdateFunction");

        generateFunction = K.function(generateInputs, new List<Tensor> { generatorOutput }, null, "GenerateFunction");
    }


    public Array GenerateBatch(Array inputConditions, Array inputNoise)
    {
        List<Array> inputLists = new List<Array>();
        if (HasConditionInput)
        {
            inputLists.Add(inputConditions);
        }
        if (HasNoiseInput)
        {
            inputLists.Add(inputNoise);
        }

        var result = generateFunction.Call(inputLists);
        return (Array)result[0].eval();

    }



}
