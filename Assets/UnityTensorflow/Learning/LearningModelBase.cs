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

public abstract class LearningModelBase : MonoBehaviour
{

    /// <summary>
    /// Whether this model has visual observation input. 
    /// </summary>
    protected bool HasVisualObservation { get; private set; }
    /// <summary>
    /// Whether this model has vector observation input. 
    /// </summary>
    protected bool HasVectorObservation { get; private set; }
    /// <summary>
    /// Whether the model uses recurrent network. Currently recurrent network is not supported.
    /// </summary>
    protected bool HasRecurrent { get; private set; } = false;

    [Tooltip("checkpoint to load if you are not using the trainer to load checkpoint")]
    public TextAsset checkpointToLoad = null;
    [Tooltip("all namescope will be under this modelName.if it is null or empty, there will be not namescope")]
    public string modelName = null;
    /// <summary>
    /// Total vector observation size, considering the stacked vector observations
    /// </summary>
    protected int StateSize { get; private set; }
    protected int[] ActionSize { get; private set; }
    protected SpaceType ActionSpace { get; private set; }
    /// <summary>
    /// Whether training is enabled in this model.
    /// </summary>
    public bool TrainingEnabled { get { return trainingEnabled; } protected set { trainingEnabled = value; } }

    [SerializeField]
    [ReadOnly]
    public bool trainingEnabled = false;

    /// <summary>
    /// Whether the model is initialized
    /// </summary>
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

        NameScope ns = null;
        if (!string.IsNullOrEmpty(modelName))
            ns = Current.K.name_scope(modelName);

        ActionSize = brainParameters.vectorActionSize;
        StateSize = brainParameters.vectorObservationSize * brainParameters.numStackedVectorObservations;
        ActionSpace = brainParameters.vectorActionSpaceType;

        //create basic inputs
        var inputStateTensor = StateSize > 0 ? UnityTFUtils.Input(new int?[] { StateSize }, name: "InputStates")[0] : null;
        HasVectorObservation = inputStateTensor != null;
        var inputVisualTensors = CreateVisualInputs(brainParameters);
        HasVisualObservation = inputVisualTensors != null;

        //create inner intialization
        InitializeInner(brainParameters, inputStateTensor, inputVisualTensors, enableTraining ? trainerParams : null);

        //test
        Debug.LogWarning("Tensorflow Graph is saved for test purpose at: SavedGraph/" + name + ".pb");
        ((UnityTFBackend)Current.K).ExportGraphDef("SavedGraph/" + name + ".pb");

        Current.K.try_initialize_variables(true);

        if (ns != null)
            ns.Dispose();

        if (checkpointToLoad != null)
        {
            RestoreCheckpoint(checkpointToLoad.bytes,true);
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
    public virtual List<Tensor> GetAllOptimizerWeights()
    {
        if (!TrainingEnabled || optimiers == null)
            return new List<Tensor>();
        List<Tensor> allWeights = new List<Tensor>();
        foreach (var o in optimiers)
        {
            allWeights.AddRange(o.Weights);
        }
        return allWeights;
    }
    

    //the default set learning rate method.
    public virtual void SetLearningRate(float lr)
    {
        if (optimiers != null && optimiers.Count > 0)
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
    /// Set all the weights of the optimziersThis will be deprecated soon.
    /// </summary>
    /// <param name="values">array of weights. might be values returned by <see cref="GetAllOptimizerWeights"/></param>
    public virtual void SetAllOptimizerWeights(List<Array> values)
    {
        int currentIndex = 0;
        foreach (var o in optimiers)
        {
            o.set_weights(values.GetRange(currentIndex, o.Weights.Count));
            currentIndex += o.Weights.Count;
        }
    }

    public virtual void SetAllOptimizerWeights(Dictionary<string, Array> values)
    {
        if (optimiers == null)
            return;
        foreach (var o in optimiers)
        {
            var optWeights = o.Weights;
            foreach (var w in optWeights)
            {
                if (!values.ContainsKey(w.name))
                {
                    Debug.LogWarning("Value of " + w.name + " can not be found in data. Set optimizer weights failed.");
                    continue;
                }

                if ((w.int_shape.Length == 0 && values[w.name].Length == 1) || w.int_shape.Aggregate((x, y) => x * y) == values[w.name].Length)
                {
                    Current.K.set_value(w, values[w.name]);
                }
                else
                {
                    Debug.LogWarning("Value of " + w.name + " does not match Tensor shape. Set optimizer weights failed.");
                    continue;
                }


            }
        }
    }
    /// <summary>
    /// Set all weigths for the model. This will be deprecated soon.
    /// </summary>
    /// <param name="values">list of arrays that are the values of each weight</param>
    public virtual void SetAllModelWeights(List<Array> values)
    {
        List<Tensor> allModelWeights = GetAllModelWeights();

        Debug.Assert(values.Count == allModelWeights.Count, "SetAllModelWeights(): Counts of input values and parameters to update do not match.");

        for (int i = 0; i < allModelWeights.Count; ++i)
        {
            Debug.Assert(values[i].GetLength().IsEqual(Mathf.Abs(allModelWeights[i].shape.Aggregate((t, s) => t * s).Value)), "Input array shape does not match the Tensor to set value");
            Current.K.set_value(allModelWeights[i], values[i]);
        }
    }

    public virtual void SetAllModelWeights(Dictionary<string, Array> values)
    {
        List<Tensor> allModelWeights = GetAllModelWeights();
        foreach (var w in allModelWeights)
        {
            if (!values.ContainsKey(w.name))
            {
                Debug.LogWarning("Value of " + w.name + " can not be found in data. Value not set.");
                continue;
            }
            Current.K.set_value(w, values[w.name]);
        }

    }

    /// <summary>
    /// Load the data s dictionary and set the model/optimizer weights using that.
    /// </summary>
    /// <param name="data"></param>
    /// <param name="modelOnly"></param>
    protected void RestoreCheckpoint(byte[] data, bool modelOnly = false)
    {
        //deserialize the data
        var mStream = new MemoryStream(data);
        var binFormatter = new BinaryFormatter();
        var deserializedData = binFormatter.Deserialize(mStream);

        if(deserializedData is Dictionary<string, Array>)
        {
            Dictionary<string, Array> dicData = deserializedData as Dictionary<string, Array>;
            SetAllModelWeights(dicData);
            if(!modelOnly)
                SetAllOptimizerWeights(dicData);
        }
        else
        {
            Debug.LogError("Not recognized datatype to restoed from");
        }
    }
    

    /// <summary>
    /// get the models all weights,including neural network's and optimziers to a byte array. 
    /// </summary>
    /// <returns>the data</returns>
    public virtual Dictionary<string, Array> SaveCheckpoint()
    {
        List<Tensor> allWeights = new List<Tensor>();
        allWeights.AddRange(GetAllModelWeights());
        allWeights.AddRange(GetAllOptimizerWeights());

        Dictionary<string, Array> saveData = new Dictionary<string, Array>();
        foreach (var w in allWeights)
        {
            if (saveData.ContainsKey(w.name))
            {
                Debug.LogWarning("tensors with the same name:" + w.name + ", ignored.");
            }
            else
            {
                object data = w.eval();

                Array flattenedData = null;
                if (data is Array)
                {
                    flattenedData = ((Array)data).DeepFlatten();
                }
                else
                {

                    flattenedData = Array.CreateInstance(data.GetType(), 1);
                    flattenedData.SetValue(data,0);
                }
                saveData[w.name] = flattenedData;
            }
        }
        /*var binFormatter = new BinaryFormatter();
        var mStream = new MemoryStream();
        binFormatter.Serialize(mStream, saveData);*/
        return saveData;
    }

}
