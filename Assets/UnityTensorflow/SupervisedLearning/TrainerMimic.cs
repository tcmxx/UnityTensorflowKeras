using Accord.Math;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using UnityEngine;
using MLAgents;
using System.Linq;

public class TrainerMimic : Trainer
{

    public AgentDependentDecision decisionToMimicRef;

    public SupervisedLearningModel modelRef;
    public TrainerParamsMimic parameters;

    public Brain BrainToTrain { get; private set; }
    StatsLogger stats;

    protected DataBuffer dataBuffer;

    [ReadOnly]
    [SerializeField]
    private int steps = 0;
    public int Steps { get { return steps; } protected set { steps = value; } }

    public bool continueFromCheckpoint = true;
    public string checkpointPath = @"Assets\testcheckpoint.bytes";

    public override void AddExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo, Dictionary<Agent,TakeActionOutput> actionOutput)
    {
        var agentList = currentInfo.Keys;
        foreach (var agent in agentList)
        {
            var agentNewInfo = newInfo[agent];

            List<ValueTuple<string, Array>> dataToAdd = new List<ValueTuple<string, Array>>();
            dataToAdd.Add(ValueTuple.Create<string, Array>("Action", actionOutput[agent].outputAction));
            if (currentInfo[agent].stackedVectorObservation.Count > 0)
                dataToAdd.Add(ValueTuple.Create<string, Array>("VectorObservation", currentInfo[agent].stackedVectorObservation.ToArray()));

            for (int i = 0; i < BrainToTrain.brainParameters.cameraResolutions.Length; ++i)
            {
                var res = BrainToTrain.brainParameters.cameraResolutions[i];
                Array arrayToAdd = TextureToArray(currentInfo[agent].visualObservations[i], res.blackAndWhite).ExpandDimensions(0);
                dataToAdd.Add(ValueTuple.Create<string, Array>("VisualObservation" + i, arrayToAdd));
            }
            dataBuffer.AddData(dataToAdd.ToArray());

        }
    }

    public override int GetMaxStep()
    {
        return parameters.maxTotalSteps;
    }

    public override int GetStep()
    {
        return Steps;
    }

    public override void IncrementStep()
    {
        Steps++;
        if (Steps % parameters.saveModelInterval == 0)
        {
            SaveModel();
            SaveTrainingData();
        }
    }

    public override void Initialize()
    {
        stats = new StatsLogger();

        modelRef.Initialize(BrainToTrain.brainParameters,isTraining);

        var brainParameters = BrainToTrain.brainParameters;

        //intialize data buffer
        List<DataBuffer.DataInfo> allBufferData = new List<DataBuffer.DataInfo>() {
            new DataBuffer.DataInfo("Action", typeof(float), new int[] { brainParameters.vectorActionSpaceType == SpaceType.continuous ? brainParameters.vectorActionSize : 1 })
        };

        if (brainParameters.vectorObservationSize > 0)
            allBufferData.Add(new DataBuffer.DataInfo("VectorObservation", typeof(float), new int[] {brainParameters.vectorObservationSize * brainParameters.numStackedVectorObservations  }));

        for (int i = 0; i < brainParameters.cameraResolutions.Length; ++i)
        {
            int width = brainParameters.cameraResolutions[i].width;
            int height = brainParameters.cameraResolutions[i].height;
            int channels;
            if (brainParameters.cameraResolutions[i].blackAndWhite)
                channels = 1;
            else
                channels = 3;

            allBufferData.Add(new DataBuffer.DataInfo("VisualObservation" + i, typeof(float), new int[] { height, width, channels }));
        }
        dataBuffer = new DataBuffer(parameters.maxBufferSize, allBufferData.ToArray());

        if (continueFromCheckpoint)
        {
            LoadModel();
            LoadTrainingData();
        }
    }

    public override bool IsReadyUpdate()
    {
        return parameters.batchSize <= dataBuffer.CurrentCount;
    }

    public override void ProcessExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo)
    {
        return;
    }

    public override void SetBrain(Brain brain)
    {
        this.BrainToTrain = brain;
    }

    public override Dictionary<Agent,TakeActionOutput> TakeAction(Dictionary<Agent, AgentInfo> agentInfos)
    {
        var result = new Dictionary<Agent, TakeActionOutput>();

        var agentList = new List<Agent>(agentInfos.Keys);

        float[,] vectorObsAll = CreateVectorIInputBatch(agentInfos, agentList);
        var visualObsAll = CreateVisualIInputBatch(agentInfos, agentList, BrainToTrain.brainParameters.cameraResolutions);

        float[,] actions = null;
        bool useHeuristic = UnityEngine.Random.Range(0, 1.0f) < parameters.chanceOfUsingheuristicForOptimization ? true : false;
        if (useHeuristic)
            actions = modelRef.EvaluateAction(vectorObsAll, visualObsAll);
        else
            actions = new float[agentList.Count, BrainToTrain.brainParameters.vectorActionSize];

        int i = 0;

        float actionDiff = 0;   //difference between decision and from networknetwork

        foreach (var agent in agentList)
        {
            var info = agentInfos[agent];

            var action = decisionToMimicRef.Decide(agent, info.stackedVectorObservation, info.visualObservations, new List<float>(actions.GetRow(i)));

            var tempAction = new TakeActionOutput();
            tempAction.outputAction = action;
            result[agent] = tempAction;


            if (useHeuristic)
            {
                actionDiff += Enumerable.Zip(action, actions.GetRow(i), (a, b) => Mathf.Sqrt((a - b)* (a - b))).Aggregate((a,v)=>a+v);
            }
            i++;
        }

        if (useHeuristic)
        {
            stats.AddData("action difference", actionDiff/ i);
        }

        return result;
    }

    public override void UpdateModel()
    {
        var fetches = new List<ValueTuple<string, int, string>>();
        if (BrainToTrain.brainParameters.vectorObservationSize > 0)
            fetches.Add(new ValueTuple<string, int, string>("VectorObservation", 0, "VectorObservation"));
        fetches.Add(new ValueTuple<string, int, string>("Action", 0, "Action"));
        for (int i = 0; i < BrainToTrain.brainParameters.cameraResolutions.Length; ++i)
        {
            fetches.Add(new ValueTuple<string, int, string>("VisualObservation" + i, 0, "VisualObservation" + i));
        }


        float loss = 0;
        for (int i = 0; i < parameters.numIterationPerTrain; ++i)
        {
            var samples = dataBuffer.RandomSample(parameters.batchSize, fetches.ToArray());

            var vectorObsArray = samples.TryGetOr("VectorObservation", null);
            float[,] vectorObservations = vectorObsArray == null ? null : vectorObsArray as float[,];
            float[,] actions = (float[,])samples["Action"];
            List<float[,,,]> visualObservations = null;
            for (int j = 0; j < BrainToTrain.brainParameters.cameraResolutions.Length; ++j)
            {
                if (j == 0)
                    visualObservations = new List<float[,,,]>();
                visualObservations.Add((float[,,,])samples["VisualObservation" + j]);
            }

            //int actionUnitSize = (BrainToTrain.brainParameters.vectorActionSpaceType == SpaceType.continuous ? BrainToTrain.brainParameters.vectorActionSize : 1);
            //int totalStateSize = BrainToTrain.brainParameters.vectorObservationSize * BrainToTrain.brainParameters.numStackedVectorObservations;

            float temoLoss = modelRef.TrainBatch(vectorObservations, visualObservations, actions);
            loss += temoLoss;
        }

        stats.AddData("loss", loss / parameters.numIterationPerTrain, parameters.lossLogInterval);
    }



    public void SaveModel()
    {
        var data = modelRef.SaveCheckpoint();
        var fullPath = Path.GetFullPath(checkpointPath);
        fullPath = fullPath.Replace('/', Path.DirectorySeparatorChar);
        fullPath = fullPath.Replace('\\', Path.DirectorySeparatorChar);
        File.WriteAllBytes(fullPath, data);
        Debug.Log("Saved model checkpoint to " + fullPath);
    }
    public void LoadModel()
    {
        string fullPath = Path.GetFullPath(checkpointPath);
        fullPath = fullPath.Replace('/', Path.DirectorySeparatorChar);
        fullPath = fullPath.Replace('\\', Path.DirectorySeparatorChar);
        if (!File.Exists(fullPath))
        {
            Debug.Log("Model checkpoint not exist at: " + fullPath);
            return;
        }
        var bytes = File.ReadAllBytes(fullPath);
        modelRef.RestoreCheckpoint(bytes);
        Debug.Log("Model loaded  from checkpoint " + fullPath);
    }

    public void SaveTrainingData()
    {
        var binFormatter = new BinaryFormatter();
        var mStream = new MemoryStream();

        binFormatter.Serialize(mStream, dataBuffer);
        var data = mStream.ToArray();

        string dir = Path.GetDirectoryName(checkpointPath);
        string file = Path.GetFileNameWithoutExtension(checkpointPath);
        string fullPath = Path.GetFullPath(Path.Combine(dir, file + "_trainingdata.bytes"));
        fullPath = fullPath.Replace('/', Path.DirectorySeparatorChar);
        fullPath = fullPath.Replace('\\', Path.DirectorySeparatorChar);

        File.WriteAllBytes(fullPath, data);
        Debug.Log("Saved training data to " + fullPath);


    }
    public void LoadTrainingData()
    {
        string dir = Path.GetDirectoryName(checkpointPath);
        string file = Path.GetFileNameWithoutExtension(checkpointPath);
        string savepath = Path.Combine(dir, file + "_trainingdata.bytes");

        string fullPath = Path.GetFullPath(savepath);

        fullPath = fullPath.Replace('/', Path.DirectorySeparatorChar);
        fullPath = fullPath.Replace('\\', Path.DirectorySeparatorChar);

        if (!File.Exists(fullPath))
        {
            Debug.Log("Training data not exist at: " + fullPath);
            return;
        }
        var bytes = File.ReadAllBytes(fullPath);

        //deserialize the data
        var mStream = new MemoryStream(bytes);
        var binFormatter = new BinaryFormatter();
        dataBuffer = (DataBuffer)binFormatter.Deserialize(mStream);

        Debug.Log("Loaded training data from " + fullPath);
    }

}
