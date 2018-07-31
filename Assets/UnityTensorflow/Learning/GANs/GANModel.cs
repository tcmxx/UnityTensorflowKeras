
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
using MLAgents;

public class GANModel : LearningModelBase, ISupervisedLearningModel
{

    [ShowAllPropertyAttr]
    public GANNetwork network;

    protected Tensor inputCorrectLabel;

    public float generatorL2LossWeight = 1;

    protected Function trainGeneratorFunction;
    protected Function trainDiscriminatorFunction;
    protected Function generateFunction;

    protected Function predictFunction;
    protected Function restoreFromPredictFunction;

    public bool HasNoiseInput { get; private set; }
    public bool HasConditionInput { get; private set; }
    public bool HasGeneratorL2Loss { get; private set; }

    public int[] outputShape;
    public int[] inputNoiseShape;
    public int[] inputConditionShape;
    public bool hasGeneratorL2Loss = false;


    public OptimizerCreator generatorOptimizer;
    public OptimizerCreator discriminatorOptimizer;

    public bool initializeOnAwake = false;

    public TrainerParamsGAN TrainerParams { get; protected set; } = null;
    
    public float GeneratorLR { get { return generatorLR; }
        set {
            generatorLR = value;
            SetLearningRate(generatorLR,0);
        }
    }
    private float generatorLR = 0.001f;

    public float DiscriminatorLR
    {
        get { return discriminatorLR; }
        set
        {
            discriminatorLR = value;
            SetLearningRate(discriminatorLR,1);
        }
    }
    private float discriminatorLR = 0.001f;


    private void Awake()
    {
        if (initializeOnAwake)
        {
            Initialize();
            //Debug.LogWarning("saved graph for test");
            //((UnityTFBackend)K).ExportGraphDef("SavedGraph/test.pb");
        }
        
    }

    /// <summary>
    /// Initialize the GAN model based on the current value fields, without considering the MLAgent stuff. 
    /// </summary>
    public void Initialize(bool enableTraining = true)
    {
        Debug.Assert(Initialized == false, "model already initialized");

        HasNoiseInput = inputNoiseShape != null && inputNoiseShape.Length > 0;
        HasConditionInput = inputConditionShape != null && inputConditionShape.Length > 0;
        HasGeneratorL2Loss = hasGeneratorL2Loss;
       

        //create generator input tensors
        Tensor inputCondition = null;
        if(HasConditionInput)
            inputCondition = UnityTFUtils.Input(inputConditionShape.Select((t)=>(int?)t).ToArray(), name:"InputConditoin")[0];
        Tensor inputNoise = null;
        if (HasNoiseInput)
            inputNoise = UnityTFUtils.Input(inputNoiseShape.Select((t) => (int?)t).ToArray(), name: "InputNoise")[0];

        Debug.Assert(HasNoiseInput || HasConditionInput, "GAN needs at least one of noise or condition input");

        Tensor inputTargetToJudge = UnityTFUtils.Input(outputShape.Select((t) => (int?)t).ToArray(), name: "InputTargetToJudge")[0];
        
        //build the network
        Tensor generatorOutput, disOutForGenerator, dicOutTarget;

        network.BuildNetwork(inputCondition, inputNoise, inputTargetToJudge, outputShape, out generatorOutput, out dicOutTarget, out disOutForGenerator);
        
        //build the loss
        //generator gan loss
        Tensor genGANLoss = K.constant(0.0f, new int[] { },DataType.Float) - K.mean(K.binary_crossentropy(disOutForGenerator, K.constant(0.0f, new int[] { }, DataType.Float), false),new int[]{ 0,1});
        Tensor genLoss = genGANLoss;
        //generator l2Loss if use it
        Tensor l2Loss = null;
        Tensor inputGeneratorTarget = null;
        Tensor inputL2LossWeight = null;
        if (hasGeneratorL2Loss)
        {
            inputL2LossWeight = UnityTFUtils.Input(batch_shape: new int?[] { }, name: "l2LossWeight", dtype: DataType.Float)[0];
            inputGeneratorTarget = UnityTFUtils.Input(outputShape.Select((t) => (int?)t).ToArray(), name: "GeneratorTarget")[0];

            int[] reduceDim = new int[outputShape.Length];
            for(int i = 0; i < reduceDim.Length; ++i)
            {
                reduceDim[i] = i;
            }
            l2Loss = K.mul(inputL2LossWeight, K.mean(new MeanSquareError().Call(inputGeneratorTarget, generatorOutput), reduceDim));
            genLoss = genGANLoss + l2Loss;
        }

        //discriminator loss
        inputCorrectLabel = UnityTFUtils.Input(new int?[] { 1 },name:"InputCorrectLabel")[0];
        Tensor discLoss = K.mean(K.binary_crossentropy(dicOutTarget, inputCorrectLabel, false),new int[] { 0,1});



        //create the Functions inputs
        List<Tensor> generatorTrainInputs = new List<Tensor>();
        List<Tensor> generateInputs = new List<Tensor>();
        List<Tensor> discriminatorTrainInputs = new List<Tensor>();
        discriminatorTrainInputs.Add(inputTargetToJudge);
        discriminatorTrainInputs.Add(inputCorrectLabel);
        if (HasConditionInput)
        {
            generatorTrainInputs.Add(inputCondition);
            generateInputs.Add(inputCondition);
            discriminatorTrainInputs.Add(inputCondition);
        }
        if(HasNoiseInput)
        {
            generatorTrainInputs.Add(inputNoise);
            generateInputs.Add(inputNoise);
        }
        if (hasGeneratorL2Loss)
        {
            generatorTrainInputs.Add(inputGeneratorTarget);
            generatorTrainInputs.Add(inputL2LossWeight);
        }

        //create optimizers
        if (enableTraining)
        {
            var generatorUpdate = AddOptimizer(network.GetGeneratorWeights(), genLoss, generatorOptimizer);
            trainGeneratorFunction = K.function(generatorTrainInputs, new List<Tensor> { genLoss }, generatorUpdate, "GeneratorUpdateFunction");

            var discriminatorUpdate = AddOptimizer(network.GetDiscriminatorWeights(), discLoss, discriminatorOptimizer);
            trainDiscriminatorFunction = K.function(discriminatorTrainInputs, new List<Tensor> { discLoss }, discriminatorUpdate, "DiscriminatorUpdateFunction");
        }
        generateFunction = K.function(generateInputs, new List<Tensor> { generatorOutput }, null, "GenerateFunction");

        //create functoin for training with prediction method
        CreateTrainWithPredictionFunctions();

        Initialized = true;
        TrainingEnabled = enableTraining;
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

    public float TrainGeneratorBatch(Array inputConditions, Array inputNoise, Array inputGeneratorTargets)
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
        if (HasGeneratorL2Loss)
        {
            inputLists.Add(inputGeneratorTargets);
            inputLists.Add(new float[] { generatorL2LossWeight });
        }

        var result = trainGeneratorFunction.Call(inputLists);
        //Array test = (Array)result[0].eval(); 
        return (float)result[0].eval();
    }

    public float TrainDiscriminatorBatch(Array inputTargetsToJudge, float[,] inputCorrectLabels,Array inputConditions)
    {
        List<Array> inputLists = new List<Array>();
        inputLists.Add(inputTargetsToJudge);
        inputLists.Add(inputCorrectLabels);

        if (HasConditionInput)
        {
            inputLists.Add(inputConditions);
        }

        var result = trainDiscriminatorFunction.Call(inputLists);
        return (float)result[0].eval();
    }


    /// <summary>
    /// Train the discriminator with correct data. And then generate the same amount of fake data use the same conditions to train the discrminator.
    /// </summary>
    /// <param name="inputTargetsToJudge"></param>
    /// <param name="inputConditions"></param>
    /// <returns></returns>
    public float TrainDiscriminatorBatch(Array inputTargetsReal,  Array inputConditions)
    {
        int batchSize = inputTargetsReal.GetLength(0);
        //generate noise
        Array noise = null;
        if (HasNoiseInput)
        {
            noise = MathUtils.GenerateWhiteNoise(batchSize, -1, 1, inputNoiseShape);
        }
        //genrate fake data
        Array generatedFake = GenerateBatch(inputConditions, noise);

        //genreate labels
        float[,] fakeLabels = new float[batchSize, 1];
        float[,] realLabels = new float[batchSize, 1];
        for (int i = 0; i < batchSize; ++i)
        {
            fakeLabels[i, 0] = 0;
            realLabels[i, 0] = 1;
        }

        //train with fake labels
        float l1 = TrainDiscriminatorBatch(generatedFake, fakeLabels, inputConditions);
        float l2 = TrainDiscriminatorBatch(inputTargetsReal, realLabels, inputConditions);
        return (l1 + l2) / 2;
    }

    public void PredictWeights()
    {
        predictFunction.Call(new List<Array>());
    }
    public void RestoreFromPredictedWeights()
    {
        restoreFromPredictFunction.Call(new List<Array>());
    }

    protected void CreateTrainWithPredictionFunctions()
    {
        List<Tensor> discWeights = network.GetDiscriminatorWeights();
        List<List<Tensor>> predictWeightsUpdate = new List<List<Tensor>>();
        List<List<Tensor>> restoreWeightsUpdate = new List<List<Tensor>>();

        using (K.name_scope("Prediction"))
        {
            IEnumerable<int?[]> shapes;
            Tensor[] saves;

            using (K.name_scope("Vars"))
            {
                shapes = discWeights.Select(p => K.get_variable_shape(p));
                saves = shapes.Select(shape => K.zeros(shape)).ToArray();
            }

            using (K.name_scope("Predicts"))
            {
                for (int i = 0; i < discWeights.Count; ++i)
                {
                    var diff = discWeights[i] - saves[i];
                    var newWeight = diff + discWeights[i];
                    predictWeightsUpdate.Add(new List<Tensor> { K.update(saves[i], discWeights[i], discWeights[i].name+"_predict") });
                    predictWeightsUpdate.Add(new List<Tensor> { K.update(discWeights[i], newWeight, discWeights[i].name + "_save") });
                }
            }

            using (K.name_scope("Restore"))
            {
                for (int i = 0; i < discWeights.Count; ++i)
                {
                    restoreWeightsUpdate.Add(new List<Tensor> { K.update(discWeights[i], saves[i], discWeights[i].name) });
                }
            }
        }

        predictFunction = K.function(null, null, predictWeightsUpdate, "PredictFunction");
        restoreFromPredictFunction = K.function(null, null, restoreWeightsUpdate, "RestoreFromPredictFunction");
    }




    public override List<Tensor> GetAllModelWeights()
    {
        List<Tensor> weights = new List<Tensor>();
        weights.AddRange(network.GetGeneratorWeights());
        weights.AddRange(network.GetDiscriminatorWeights());
        return weights;
    }



    public override void InitializeInner(BrainParameters brainParameters, Tensor stateTensor, List<Tensor> visualTensors, TrainerParams trainerParams)
    {
        if (brainParameters.cameraResolutions != null && brainParameters.cameraResolutions.Length != 0)
        {
            Debug.LogError("GAN for ML agent does not support visual input yet");
        }
        Debug.Assert(brainParameters.vectorActionSpaceType == SpaceType.continuous, "GAN for ML agent does not support discrete action space.");

        TrainerParams = trainerParams as TrainerParamsGAN;
        if (trainerParams != null && TrainerParams == null)
        {
            Debug.LogError("Trainer params for GAN needs to be a TrainerParamsGAN type");
        }

        outputShape = new int[] { ActionSize };
        inputConditionShape = new int[] { StateSize };

        Initialize(trainerParams != null);
    }



    /// <summary>
    /// use for ML agent TrainerMimic
    /// </summary>
    /// <param name="vectorObservation"></param>
    /// <param name="visualObservation"></param>
    /// <returns></returns>
    public float[,] EvaluateAction(float[,] vectorObservation, List<float[,,,]> visualObservation)
    {
        return (float[,])GenerateBatch(vectorObservation, MathUtils.GenerateWhiteNoise(vectorObservation.GetLength(0), -1, 1,inputNoiseShape));
    }

    /// <summary>
    /// use for ML agent TrainerMimic
    /// </summary>
    /// <param name="vectorObservations"></param>
    /// <param name="visualObservations"></param>
    /// <param name="actions"></param>
    /// <returns></returns>
    public float TrainBatch(float[,] vectorObservations, List<float[,,,]> visualObservations, float[,] actions)
    {
        float disLoss = 0, genLoss = 0;
        int batchSize = vectorObservations.GetLength(0);
        for (int i = 0; i < TrainerParams.discriminatorTrainCount; ++i)
        {
            disLoss += TrainDiscriminatorBatch(actions, vectorObservations);
        }

        if(TrainerParams.usePrediction)
            PredictWeights();

        Array noise = null;
        if (HasNoiseInput)
        {
            noise = MathUtils.GenerateWhiteNoise(batchSize, -1, 1, inputNoiseShape);
        }
        for (int i = 0; i < TrainerParams.generatorTrainCount; ++i)
        {
            genLoss += TrainGeneratorBatch(vectorObservations, noise, actions);
        }

        if (TrainerParams.usePrediction)
            RestoreFromPredictedWeights();

        return (disLoss / TrainerParams.discriminatorTrainCount +  genLoss / TrainerParams.generatorTrainCount)/2;
    }

    public override void SetLearningRate(float lr)
    {
        SetLearningRate(lr,0);
        SetLearningRate(lr, 1);
    }
}
