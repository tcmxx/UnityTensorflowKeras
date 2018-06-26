using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using Accord.Math;
using System.Linq;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
#if UNITY_EDITOR
using UnityEditor;
#endif
using static Current;


/// <summary>
/// actor critic network abstract class
/// </summary>
public class SupervisedLearningModel : MonoBehaviour
{
    public int StateSize { get; private set; }
    public int ActionSize { get; private set; }
    public SpaceType ActionSpace { get; private set; }

    public Adam optimizer;
    public SupervisedLearningNetwork network;

    public Function ActionFunction { get; private set; }
    public Function UpdateFunction { get; private set; }

    public bool HasVisualObservation { get; private set; }
    public bool HasVectorObservation { get; private set; }
    public bool HasRecurrent { get; private set; } = false;

    public virtual void Initialize(Brain brain)
    {
        ActionSize = brain.brainParameters.vectorActionSize;
        StateSize = brain.brainParameters.vectorObservationSize * brain.brainParameters.numStackedVectorObservations;
        ActionSpace = brain.brainParameters.vectorActionSpaceType;

        //create basic inputs
        var inputStateTensor = StateSize > 0 ? UnityTFUtils.Input(new int?[] { StateSize }, name: "InputStates")[0] : null;
        HasVectorObservation = inputStateTensor != null;
        var inputVisualTensors = CreateVisualInputs(brain);
        HasVisualObservation = inputVisualTensors != null;


        //build the network
        Tensor outputAction = network.BuildNetwork(inputStateTensor, inputVisualTensors, null, ActionSize, ActionSpace);

        //training inputs
        var inputActionLabel = UnityTFUtils.Input(new int?[] { ActionSpace == SpaceType.continuous ? ActionSize : 1 }, name: "InputAction", dtype: ActionSpace == SpaceType.continuous ? DataType.Float : DataType.Int32)[0];
        //creat the loss
        Tensor loss = null;
        if (ActionSpace == SpaceType.discrete)
        {
            Tensor actionOnehot = K.one_hot(inputActionLabel, K.constant(ActionSize, dtype: DataType.Int32), K.constant(1.0f), K.constant(0.0f));
            loss = K.mean(K.categorical_crossentropy(actionOnehot, outputAction, false));
        }
        else
        {
            loss = K.mean(new MeanSquareError().Call(inputActionLabel, outputAction));
        }


        //add inputs, outputs and parameters to the list
        List<Tensor> updateParameters = GetAllModelWeights();
        List<Tensor> allInputs = new List<Tensor>();
        List<Tensor> observationInputs = new List<Tensor>();

        if (HasVectorObservation)
        {
            allInputs.Add(inputStateTensor);
            observationInputs.Add(inputStateTensor);
        }
        if (HasVisualObservation)
        {
            allInputs.AddRange(inputVisualTensors);
            observationInputs.AddRange(inputVisualTensors);
        }
        allInputs.Add(inputActionLabel);
        //create optimizer and create necessary functions
        optimizer = new Adam(lr: 0.001);
        var updates = optimizer.get_updates(updateParameters, null, loss); ;
        UpdateFunction = K.function(allInputs, new List<Tensor> { loss }, updates, "UpdateFunction");
        ActionFunction = K.function(observationInputs, new List<Tensor> { outputAction }, null, "ActionFunction");



        //test
        Debug.LogWarning("Tensorflow Graph is saved for test purpose at: SavedGraph/PPOTest.pb");
        ((UnityTFBackend)K).ExportGraphDef("SavedGraph/SuperviseTest.pb");

    }



    /// <summary>
    /// 
    /// </summary>
    /// <param name="vectorObservation"></param>
    /// <param name="visualObservation"></param>
    /// <returns></returns>
    public virtual float[,] EvaluateAction(float[,] vectorObservation, List<float[,,,]> visualObservation)
    {
        List<Array> inputLists = new List<Array>();
        if (HasVectorObservation)
        {
            Debug.Assert(vectorObservation != null, "Must Have vector observation inputs!");
            inputLists.Add(vectorObservation);
        }
        if (HasVisualObservation)
        {
            Debug.Assert(visualObservation != null, "Must Have visual observation inputs!");
            inputLists.AddRange(visualObservation);
        }

        var result = ActionFunction.Call(inputLists);

        var outputAction = ((float[,])result[0].eval());

        float[,] actions = new float[outputAction.GetLength(0), ActionSpace == SpaceType.continuous ? outputAction.GetLength(1) : 1];
        if (ActionSpace == SpaceType.continuous)
        {
            for (int j = 0; j < outputAction.GetLength(0); ++j)
            {
                for (int i = 0; i < outputAction.GetLength(1); ++i)
                {
                    actions[j, i] = outputAction[j, i];
                }
            }
        }
        else if (ActionSpace == SpaceType.discrete)
        {
            for (int j = 0; j < outputAction.GetLength(0); ++j)
            {
                actions[j, 0] = outputAction.GetRow(j).ArgMax();
            }
        }

        return actions;
    }


    public virtual float TrainBatch(float[,] vectorObservations, List<float[,,,]> visualObservations, float[,] actions)
    {
        List<Array> inputs = new List<Array>();
        if (vectorObservations != null)
            inputs.Add(vectorObservations);
        if (visualObservations != null)
            inputs.AddRange(visualObservations);
        if (ActionSpace == SpaceType.continuous)
            inputs.Add(actions);
        else if (ActionSpace == SpaceType.discrete)
        {
            int[,] actionsInt = actions.Convert(t => Mathf.RoundToInt(t));
            inputs.Add(actionsInt);
        }

        var loss = UpdateFunction.Call(inputs);
        var result = (float)loss[0].eval() ;

        return result;
    }


    protected List<Tensor> CreateVisualInputs(Brain brain)
    {
        if (brain.brainParameters.cameraResolutions == null || brain.brainParameters.cameraResolutions.Length == 0)
        {
            return null;
        }
        List<Tensor> allInputs = new List<Tensor>();
        int i = 0;
        foreach (var r in brain.brainParameters.cameraResolutions)
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
        SetAllOptimizerWeights(arrayData.GetRange(modelWeigthLength, optimizerWeightLength));
    }

    public virtual List<Tensor> GetAllModelWeights()
    {
        List<Tensor> parameters = new List<Tensor>();
        parameters.AddRange(network.GetWeights());
        return parameters;
    }
    public virtual List<Array> GetAllOptimizerWeights()
    {
        return optimizer.get_weights();
    }

    public virtual void SetAllModelWeights(List<Array> values)
    {
        List<Tensor> updateParameters = new List<Tensor>();
        updateParameters.AddRange(network.GetWeights());

        Debug.Assert(values.Count == updateParameters.Count, "Counts of input values and parameters to update do not match.");

        for (int i = 0; i < updateParameters.Count; ++i)
        {
            Debug.Assert(values[i].GetLength().IsEqual(Mathf.Abs(updateParameters[i].shape.Aggregate((t, s) => t * s).Value)), "Input array shape does not match the Tensor to set value");
            K.set_value(updateParameters[i], values[i]);
        }
    }
    public virtual void SetAllOptimizerWeights(List<Array> values)
    {
        optimizer.set_weights(values);
    }








    public virtual void OnInspector()
    {
#if UNITY_EDITOR
        Editor.CreateEditor(this).OnInspectorGUI();
#endif
    }
}
