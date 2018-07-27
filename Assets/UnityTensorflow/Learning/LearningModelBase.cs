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

        //create functions for evaluation
        List<Tensor> observationInputs = new List<Tensor>();

        if (HasVectorObservation)
        {
            observationInputs.Add(inputStateTensor);
        }
        if (HasVisualObservation)
        {
            observationInputs.AddRange(inputVisualTensors);
        }


        InitializeInner(brainParameters, inputStateTensor, inputVisualTensors, observationInputs, enableTraining? trainerParams:null);

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

    public abstract void InitializeInner(BrainParameters brainParameters, Tensor stateTensor, List<Tensor> visualTensors, List<Tensor> allobservationInputs, TrainerParams trainerParams);


    public List<List<Tensor>> AddOptimizer(List<Tensor> allWeights, Tensor loss, OptimizerCreator optimizerCreator) 
    {
        if (optimiers == null)
            optimiers = new List<OptimizerBase>();
        var newOpt = optimizerCreator.CreateOptimizer();
        optimiers.Add(newOpt);
        return newOpt.get_updates(allWeights, null, loss); ;
    } 

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
    public virtual void SetAllOptimizerWeights(List<Array> values)
    {
        int currentIndex = 0;
        foreach(var o in optimiers)
        {
            o.set_weights(values.GetRange(currentIndex, o.Weights.Count));
            currentIndex += o.Weights.Count;
        }
    }

    public virtual void SetLearningRate(float lr, int optimierIndex = 0)
    {
        Debug.Assert(TrainingEnabled == true, "The model needs to initalized with Training enabled to use SetLearningRate()");
        optimiers[optimierIndex].SetLearningRate(lr);
    }


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



    public abstract List<Tensor> GetAllModelWeights();

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



    public virtual void RestoreCheckpoint(byte[] data)
    {
        //deserialize the data
        var mStream = new MemoryStream(data);
        var binFormatter = new BinaryFormatter();
        var floatData = (List<float[]>)binFormatter.Deserialize(mStream);

        List<Array> arrayData = floatData.ConvertAll(t => (Array)t);
        var optimizerWeightLength = GetAllOptimizerWeights().Count;   //used for initialize the graph.
        var modelWeigthLength = GetAllModelWeights().Count;      //get the length of model weights and training param weights
        SetAllModelWeights(arrayData.GetRange(0, modelWeigthLength));

        if (arrayData.Count >= modelWeigthLength + optimizerWeightLength && optimizerWeightLength > 0)
        {
            SetAllOptimizerWeights(arrayData.GetRange(modelWeigthLength, optimizerWeightLength));
        }
    }

    /// <summary>
    /// save the models all parameters to a byte array
    /// </summary>
    /// <returns></returns>
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
