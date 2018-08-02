using Accord.Math;
using KerasSharp.Backends;
using KerasSharp.Engine.Topology;
using KerasSharp.Models;
using KerasSharp.Optimizers;
using MLAgents;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

public abstract class LearningModelBase : MonoBehaviour {

    protected bool HasVisualObservation { get;  set; }
    protected bool HasVectorObservation { get;  set; }
    protected bool HasRecurrent { get;  set; } = false;

    public TextAsset checkpointTOLoad = null;

    protected int StateSize { get;  set; }
    protected int ActionSize { get;  set; }
    protected SpaceType ActionSpace { get;  set; }

    public bool TrainingEnabled { get { return trainingEnabled; } protected set { trainingEnabled = value; } }
    [SerializeField]
    [ReadOnly]
    public bool trainingEnabled = false;

    public bool Initialized { get; protected set; } = false;

    protected List<OptimizerBase> optimiers;

    /// <summary>
    /// implement this method for your learning model for use of ML agent. It is called by a Trainer. You should create everything inccluding the neural network and optimizer(if trainer params if not null),
    /// using the inputs tensors
    /// </summary>
    /// <param name="brainParameters">brain parameter of the MLagent brain</param>
    /// <param name="stateTensor">the input tensor of the vector observation</param>
    /// <param name="visualTensors">input tensors of visual observations</param>
    /// <param name="trainerParams">trainer parameters passed by the trainer. You if it is null, training is no enbled and you dont have to implement the optimzing parts. </param>
    public abstract void InitializeInner(BrainParameters brainParameters, Tensor stateTensor, List<Tensor> visualTensors, TrainerParams trainerParams);

    /// <summary>
    /// Implement this method for getting all model's weights (not including optimizers parameters), for save and restore purpose.
    /// </summary>
    /// <returns>list of all model's weights</returns>
    public abstract List<Tensor> GetAllModelWeights();


    /// <summary>
    /// Trainers will call this method to initialize the model. This method will call the InitializeInner()
    /// </summary>
    /// <param name="brainParameters">brain parameter of the MLagent brain</param>
    /// <param name="enableTraining">whether enable training</param>
    /// <param name="trainerParams">trainer parameters passed by the trainer. </param>
    public virtual void Initialize(BrainParameters brainParameters, bool enableTraining, TrainerParams trainerParams)
    {
        Debug.Assert(Initialized == false, "Model already Initalized");

        ActionSize = brainParameters.vectorActionSize;
        StateSize = brainParameters.vectorObservationSize * brainParameters.numStackedVectorObservations;
        ActionSpace = brainParameters.vectorActionSpaceType;

        //create basic inputs
        var inputStateTensor = StateSize > 0 ? UnityTFUtils.Input(new int?[] { StateSize }, name: "InputStates")[0] : null;
        HasVectorObservation = inputStateTensor != null;
        var inputVisualTensors = CreateVisualInputs(brainParameters);
        HasVisualObservation = inputVisualTensors != null;

        //create inner intialization
        InitializeInner(brainParameters, inputStateTensor, inputVisualTensors,  enableTraining? trainerParams:null);

        //test
        //Debug.LogWarning("Tensorflow Graph is saved for test purpose at: SavedGraph/PPOTest.pb");
        //((UnityTFBackend)Current.K).ExportGraphDef("SavedGraph/"+name+".pb");

        Current.K.try_initialize_variables();
        if (checkpointTOLoad != null)
        {
            RestoreCheckpoint(checkpointTOLoad.bytes);
        }
        Initialized = true;
        TrainingEnabled = enableTraining;



    }

    /// <summary>
    /// Add a optimizer to the model. the new optimzier will be append to the optimiers list.
    /// </summary>
    /// <param name="allWeights">all weights that this optimizer need to optimzer</param>
    /// <param name="loss">loss tensor</param>
    /// <param name="optimizerCreator">A OptimizerCreator object where the information of the optimizer is specified.</param>
    /// <returns></returns>
    public List<List<Tensor>> AddOptimizer(List<Tensor> allWeights, Tensor loss, OptimizerCreator optimizerCreator) 
    {
        if (optimiers == null)
            optimiers = new List<OptimizerBase>();
        var newOpt = optimizerCreator.CreateOptimizer();
        optimiers.Add(newOpt);
        return newOpt.get_updates(allWeights, null, loss); ;
    }

    /// <summary>
    /// Add a optimizer to the model. the new optimzier will be append to the optimiers list.
    /// </summary>
    /// <param name="allWeights">all weights that this optimizer need to optimzer</param>
    /// <param name="loss">loss tensor</param>
    /// <param name="optimizer">optimizer</param>
    /// <returns></returns>
    public List<List<Tensor>> AddOptimizer(List<Tensor> allWeights, Tensor loss, OptimizerBase optimizer)
    {
        if (optimiers == null)
            optimiers = new List<OptimizerBase>();
        optimiers.Add(optimizer);
        return optimizer.get_updates(allWeights, null, loss); ;
    }

    /// <summary>
    /// This method will return all weights used in all optimizers
    /// </summary>
    /// <returns>all weights used by optimizers</returns>
    public virtual List<Array> GetAllOptimizerWeights()
    {
        if (!TrainingEnabled)
            return new List<Array>();
        List<Array> allWeights = new List<Array>();
        foreach(var o in optimiers)
        {
            allWeights.AddRange(o.get_weights());
        }
        return allWeights;
    }

    /// <summary>
    /// Set all the weights of the optimziers
    /// </summary>
    /// <param name="values">array of weights. might be values returned by <see cref="GetAllOptimizerWeights"/></param>
    public virtual void SetAllOptimizerWeights(List<Array> values)
    {
        int currentIndex = 0;
        foreach(var o in optimiers)
        {
            o.set_weights(values.GetRange(currentIndex, o.Weights.Count));
            currentIndex += o.Weights.Count;
        }
    }

    //the default set learning rate method.
    public virtual void SetLearningRate(float lr)
    {
        SetLearningRate(lr, 0);
    }


    //set the learning weights of certain optimizer
    public virtual void SetLearningRate(float lr, int optimierIndex)
    {
        Debug.Assert(TrainingEnabled == true, "The model needs to initalized with Training enabled to use SetLearningRate()");
        optimiers[optimierIndex].SetLearningRate(lr);
    }

    /// <summary>
    /// create visual input  tensors for the BrainParameters for MLagent.
    /// </summary>
    /// <param name="brainParameters">BrainParameters</param>
    /// <returns>List of all input visual tensors</returns>
    protected static List<Tensor> CreateVisualInputs(BrainParameters brainParameters)
    {
        if (brainParameters.cameraResolutions == null || brainParameters.cameraResolutions.Length == 0)
        {
            return null;
        }
        List<Tensor> allInputs = new List<Tensor>();
        int i = 0;
        foreach (var r in brainParameters.cameraResolutions)
        {
            int width = r.width;
            int height = r.height;
            int channels;
            if (r.blackAndWhite)
                channels = 1;
            else
                channels = 3;

            var input = UnityTFUtils.Input(new int?[] { height, width, channels }, name: "InputVisual" + i)[0];
            allInputs.Add(input);

            i++;
        }

        return allInputs;
    }

    /// <summary>
    /// Set all weigths for the model.
    /// </summary>
    /// <param name="values">list of arrays that are the values of each weight</param>
    public virtual void SetAllModelWeights(List<Array> values)
    {
        List<Tensor> updateParameters = GetAllModelWeights();

        Debug.Assert(values.Count == updateParameters.Count, "SetAllModelWeights(): Counts of input values and parameters to update do not match.");

        for (int i = 0; i < updateParameters.Count; ++i)
        {
            Debug.Assert(values[i].GetLength().IsEqual(Mathf.Abs(updateParameters[i].shape.Aggregate((t, s) => t * s).Value)), "Input array shape does not match the Tensor to set value");
            Current.K.set_value(updateParameters[i], values[i]);
        }
    }

    /// <summary>
    /// set all weights including optimzierss and network's from a input byte array. 
    /// </summary>
    /// <param name="data">checkpoint data. It should be the one that is obtained from <see cref="SaveCheckpoint"/>)</param>
    public virtual void RestoreCheckpoint(byte[] data)
    {
        //deserialize the data
        var mStream = new MemoryStream(data);
        var binFormatter = new BinaryFormatter();
        var floatData = (List<float[]>)binFormatter.Deserialize(mStream);

        List<Array> arrayData = floatData.ConvertAll(t => (Array)t);
        var optimizerWeightLength = GetAllOptimizerWeights().Count;   //used for initialize the graph.
        var modelWeigthLength = GetAllModelWeights().Count;      //get the length of model weights and training param weights

        if (arrayData.Count == modelWeigthLength || arrayData.Count == modelWeigthLength + optimizerWeightLength)
        {

            SetAllModelWeights(arrayData.GetRange(0, modelWeigthLength));
            if (arrayData.Count == modelWeigthLength + optimizerWeightLength)
            {
                SetAllOptimizerWeights(arrayData.GetRange(modelWeigthLength, optimizerWeightLength));
            }
        }
        else
        {
            Debug.LogError("Saved data to load not match the model!");
        }
    }

    /// <summary>
    /// get the models all weights,including neural network's and optimziers to a byte array. 
    /// </summary>
    /// <returns>the data</returns>
    public virtual byte[] SaveCheckpoint()
    {
        List<Array> data = GetAllModelWeights().Select(t => (Array)t.eval()).ToList();
        data.AddRange(GetAllOptimizerWeights());

        List<float[]> flattenedData = new List<float[]>();
        foreach (var d in data)
        {
            flattenedData.Add(d.FlattenAndConvertArray<float>());
        }

        var binFormatter = new BinaryFormatter();
        var mStream = new MemoryStream();
        binFormatter.Serialize(mStream, flattenedData);
        return mStream.ToArray();
    }

}
