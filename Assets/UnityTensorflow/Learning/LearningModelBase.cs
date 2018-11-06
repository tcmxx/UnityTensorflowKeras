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
using KerasSharp;
using static KerasSharp.Backends.Current;
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
    [Tooltip("all namescope will be under this modelName.if it is null or empty, there will be not namescope.")]
    public string modelName = null;
    [Tooltip("what is used as the key in dictionary when saving/loading the model weights. modelName will be the prefix when using WeightSaveKeyMode.UseWeightOrder.")]
    public WeightSaveKeyMode weightSaveMode = WeightSaveKeyMode.UseTensorName;
    public enum WeightSaveKeyMode
    {
        UseTensorName,
        UseWeightOrder
    }
    protected readonly string ModelWeightPrefix = "Weight";
    protected readonly string OptimizerWeightPrefix = "OptimizerWeight";

    /// <summary>
    /// Total vector observation size, considering the stacked vector observations
    /// </summary>
    protected int StateSize { get; private set; }
    protected int[] ActionSizes { get; private set; }
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
    /// <param name="trainerParams">trainer parameters passed by the trainer. Training will not be enabled </param>
    public virtual void Initialize(BrainParameters brainParameters, bool enableTraining, TrainerParams trainerParams = null)
    {
        Debug.Assert(Initialized == false, "Model already Initalized");

        NameScope ns = null;
        if (!string.IsNullOrEmpty(modelName))
            ns = Current.K.name_scope(modelName);

        ActionSizes = brainParameters.vectorActionSize;
        StateSize = brainParameters.vectorObservationSize * brainParameters.numStackedVectorObservations;
        ActionSpace = brainParameters.vectorActionSpaceType;

        Debug.Assert(ActionSizes[0] > 0, "Action size can not be zero");

        //create basic inputs
        var inputStateTensor = StateSize > 0 ? UnityTFUtils.Input(new int?[] { StateSize }, name: "InputStates")[0] : null;
        HasVectorObservation = inputStateTensor != null;
        var inputVisualTensors = CreateVisualInputs(brainParameters);
        HasVisualObservation = inputVisualTensors != null;

        //create inner intialization
        InitializeInner(brainParameters, inputStateTensor, inputVisualTensors, enableTraining ? trainerParams : null);

        //test
        //Debug.LogWarning("Tensorflow Graph is saved for test purpose at: SavedGraph/" + name + ".pb");
        //((UnityTFBackend)Current.K).ExportGraphDef("SavedGraph/" + name + ".pb");

        Current.K.try_initialize_variables(true);

        if (ns != null)
            ns.Dispose();

        if (checkpointToLoad != null)
        {
            RestoreCheckpoint(checkpointToLoad.bytes, true);
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
    /// create the network layers that masks the logits, normalize it and output the action based on multinomial distribution
    /// </summary>
    /// <param name="all_logits"></param>
    /// <param name="action_masks"></param>
    /// <param name="outputs"></param>
    /// <param name="normalizedLogProbs"></param>
    public static void CreateDiscreteActionMaskingLayer(Tensor[] all_logits, Tensor[] action_masks, out Tensor[] outputs, out Tensor[] normalizedLogProbs)
    {

        var rawProbs = all_logits.Select((x, i) => K.softmax(x) * action_masks[i]);
        var normalizedProbs = rawProbs.Select((x) => x / (K.sum(x, 1, true) + 0.00000000001f));
        normalizedLogProbs = normalizedProbs.Select((x) => K.log(x + 0.00000000001f)).ToList().ToArray();
        outputs = normalizedLogProbs.Select(x => K.multinomial(x, K.constant(1, dtype: DataType.Int32))).ToList().ToArray();
    }

    /// <summary>
    /// Create masks that are all one for discrete action
    /// </summary>
    /// <param name="actionSizes"></param>
    /// <param name="batchSize"></param>
    /// <returns></returns>
    public static List<float[,]> CreateDummyMasks(int[] actionSizes, int batchSize)
    {
        int branchSize = actionSizes.Length;
        List<float[,]>masks = new List<float[,]>();

        for (int i = 0; i < branchSize; ++i)
        {
            int actSize = actionSizes[i];
            var branchMask = new float[batchSize, actionSizes[i]];
            for (int b = 0; b < batchSize; ++b)
            {
                for (int c = 0; c < actSize; ++c)
                {
                    branchMask[b, c] = 1;
                }
            }
            masks.Add(branchMask);
        }

        return masks;
    }



    public virtual void SetAllOptimizerWeights(Dictionary<string, Array> values)
    {
        if (optimiers == null)
            return;
        List<Tensor> allOptimizerWeights = GetAllOptimizerWeights();
        foreach (var w in allOptimizerWeights)
        {
            string saveKey = GetWeightSaveName(w, new List<Tensor>(), allOptimizerWeights);
            if (!values.ContainsKey(saveKey))
            {
                Debug.LogWarning("Value of " + saveKey + " can not be found in data. Value not set.");
                continue;
            }

            if ((w.shape.Length == 0 && values[saveKey].Length == 1) || w.shape.Aggregate((x, y) => x * y) == values[saveKey].Length)
            {
                Current.K.set_value(w, values[saveKey]);
            }
            else
            {
                Debug.LogWarning("Value of " + saveKey + " does not match Tensor shape. Set optimizer weights failed.");
                continue;
            }
            //Current.K.set_value(w, values[saveKey]);
        }
    }


    public virtual void SetAllModelWeights(Dictionary<string, Array> values)
    {
        List<Tensor> allModelWeights = GetAllModelWeights();
        foreach (var w in allModelWeights)
        {
            string saveKey = GetWeightSaveName(w, allModelWeights, new List<Tensor>());
            if (!values.ContainsKey(saveKey))
            {
                Debug.LogWarning("Value of " + saveKey + " can not be found in data. Value not set.");
                continue;
            }

            if ((w.shape.Length == 0 && values[saveKey].Length == 1) || w.shape.Aggregate((x, y) => x * y) == values[saveKey].Length)
            {
                Current.K.set_value(w, values[saveKey]);
            }
            else
            {
                Debug.LogWarning("Value of " + saveKey + " does not match Tensor shape. Set model weights failed.");
                continue;
            }
            //Current.K.set_value(w, values[saveKey]);
        }

    }

    /// <summary>
    /// Load the data s dictionary and set the model/optimizer weights using that.
    /// </summary>
    /// <param name="data"></param>
    /// <param name="modelOnly">if model only is true, optimzer data will not be loaded</param>
    public void RestoreCheckpoint(byte[] data, bool modelOnly = false)
    {
        //deserialize the data
        var mStream = new MemoryStream(data);
        var binFormatter = new BinaryFormatter();
        var deserializedData = binFormatter.Deserialize(mStream);

        if (deserializedData is Dictionary<string, Array>)
        {
            Dictionary<string, Array> dicData = deserializedData as Dictionary<string, Array>;
            SetAllModelWeights(dicData);
            if (!modelOnly)
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
        List<Tensor> modelWeights = GetAllModelWeights();
        List<Tensor> optimizerWeights = GetAllOptimizerWeights();
        allWeights.AddRange(modelWeights);
        allWeights.AddRange(optimizerWeights);

        Dictionary<string, Array> saveData = new Dictionary<string, Array>();
        foreach (var w in allWeights)
        {
            string saveKey = GetWeightSaveName(w, modelWeights, optimizerWeights);
            if (saveData.ContainsKey(saveKey))
            {
                Debug.LogWarning("tensors with the same save name:" + saveKey + ", ignored.");
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
                    flattenedData.SetValue(data, 0);
                }
                saveData[saveKey] = flattenedData;
            }
        }
        /*var binFormatter = new BinaryFormatter();
        var mStream = new MemoryStream();
        binFormatter.Serialize(mStream, saveData);*/
        return saveData;
    }


    protected string GetWeightSaveName(Tensor weight, List<Tensor> modelGroup, List<Tensor> optimizerGroup)
    {
        if (weightSaveMode == WeightSaveKeyMode.UseTensorName)
        {
            return weight.name;
        }
        else
        {
            if (modelGroup.Contains(weight))
            {
                return ModelWeightPrefix + modelGroup.IndexOf(weight);
            }
            else if (optimizerGroup.Contains(weight))
            {
                return modelName + "/" + OptimizerWeightPrefix + optimizerGroup.IndexOf(weight);
            }
            else
            {
                Debug.LogError("weight not exist in anygroup");
                return null;
            }

        }
    }
}
