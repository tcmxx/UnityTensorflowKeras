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
    
    

    [Tooltip("Whether collect data from Decision for supervised learning?")]
    public bool isCollectingData = true;
    public bool loadTrainingDataFromCheckpoint = true;
    public bool saveTrainingData = true;
    public string trainingDataSaveFileName = @"trainingData.bytes";
    StatsLogger stats;

    protected DataBuffer dataBuffer;

    [ReadOnly]
    [SerializeField]
    protected int dataBufferCount;


    protected ISupervisedLearningModel modelSL;
    protected TrainerParamsMimic parametersMimic;

    public override void Initialize()
    {
        modelSL = modelRef as ISupervisedLearningModel;
        Debug.Assert(modelSL != null, "Please assign a ISupervisedLearningModel to modelRef");
        Debug.Assert(BrainToTrain != null, "brain can not be null");
        parametersMimic = parameters as TrainerParamsMimic;
        Debug.Assert(parametersMimic != null, "Please Specify PPO Trainer Parameters");
        stats = new StatsLogger();
        modelRef.Initialize(BrainToTrain.brainParameters, isTraining, parameters);

        var brainParameters = BrainToTrain.brainParameters;

        //intialize data buffer
        Debug.Assert(brainParameters.vectorActionSize.Length <= 1, "Action branching is not supported yet");
        List<DataBuffer.DataInfo> allBufferData = new List<DataBuffer.DataInfo>() {
            new DataBuffer.DataInfo("Action", typeof(float), new int[] { brainParameters.vectorActionSpaceType == SpaceType.continuous ? brainParameters.vectorActionSize[0] : 1 })
        };

        if (brainParameters.vectorObservationSize > 0)
            allBufferData.Add(new DataBuffer.DataInfo("VectorObservation", typeof(float), new int[] { brainParameters.vectorObservationSize * brainParameters.numStackedVectorObservations }));

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
        allBufferData.Add(new DataBuffer.DataInfo("Reward", typeof(float), new int[] { 1 }));

        dataBuffer = new DataBuffer(parametersMimic.maxBufferSize, allBufferData.ToArray());

        if (continueFromCheckpoint)
        {
            LoadModel();
            
        }
        if (loadTrainingDataFromCheckpoint)
        {
            LoadTrainingData();
        }
    }

    public override void AddExperience(Dictionary<Agent, AgentInfoInternal> currentInfo, Dictionary<Agent, AgentInfoInternal> newInfo, Dictionary<Agent,TakeActionOutput> actionOutput)
    {
        var agentList = currentInfo.Keys;
        foreach (var agent in agentList)
        {
            var agentDecision = agent.GetComponent<AgentDependentDecision>();
            //only add the data to databuffer if this agent uses decision
            if (agentDecision != null && agentDecision.useDecision && isCollectingData)
            {
                var agentNewInfo = newInfo[agent];

                List<ValueTuple<string, Array>> dataToAdd = new List<ValueTuple<string, Array>>();
                dataToAdd.Add(ValueTuple.Create<string, Array>("Action", actionOutput[agent].outputAction));
                if (currentInfo[agent].stackedVectorObservation.Count > 0)
                    dataToAdd.Add(ValueTuple.Create<string, Array>("VectorObservation", currentInfo[agent].stackedVectorObservation.ToArray()));

                for (int i = 0; i < BrainToTrain.brainParameters.cameraResolutions.Length; ++i)
                {
                    var res = BrainToTrain.brainParameters.cameraResolutions[i];
                    Array arrayToAdd = currentInfo[agent].visualObservations[i].ExpandDimensions(0);
                    dataToAdd.Add(ValueTuple.Create<string, Array>("VisualObservation" + i, arrayToAdd));
                }
                dataToAdd.Add(ValueTuple.Create<string, Array>("Reward", new float[] { newInfo[agent].reward}));

                dataBuffer.AddData(dataToAdd.ToArray());
            }
        }
    }

    public override void IncrementStep()
    {
        base.IncrementStep();
        dataBufferCount = dataBuffer.CurrentCount;
        if (saveTrainingData && GetStep() % parametersMimic.saveModelInterval == 0 && GetStep() != 0)
        {
            SaveTrainingData();
        }

        if (GetStep() % parametersMimic.logInterval == 0 && GetStep() != 0)
        {
            stats.LogAllCurrentData(GetStep());
        }
    }



    public override bool IsReadyUpdate()
    {
        return parametersMimic.requiredDataBeforeTraining <= dataBuffer.CurrentCount;
    }

    public override void ProcessExperience(Dictionary<Agent, AgentInfoInternal> currentInfo, Dictionary<Agent, AgentInfoInternal> newInfo)
    {
        return;
    }



    public override Dictionary<Agent,TakeActionOutput> TakeAction(Dictionary<Agent, AgentInfoInternal> agentInfos)
    {
        var result = new Dictionary<Agent, TakeActionOutput>();

        var agentList = new List<Agent>(agentInfos.Keys);
        if (agentList.Count <= 0)
            return result;
        float[,] vectorObsAll = CreateVectorInputBatch(agentInfos, agentList);
        var visualObsAll = CreateVisualInputBatch(agentInfos, agentList, BrainToTrain.brainParameters.cameraResolutions);

        float[,] actions = null;
        var evalOutput = modelSL.EvaluateAction(vectorObsAll, visualObsAll);
        actions = evalOutput.Item1;
        var vars = evalOutput.Item2;

        int i = 0;
        int agentNumWithDecision = 0;
        float actionDiff = 0;   //difference between decision and from networknetwork

        foreach (var agent in agentList)
        {
            var info = agentInfos[agent];
            var agentDecision = agent.GetComponent<AgentDependentDecision>();

            if (agentDecision != null && agentDecision.useDecision)
            {
                //if this agent will use the decision, use it
                float[] action = null;
                if(vars != null)
                    action = agentDecision.Decide(info.stackedVectorObservation, info.visualObservations, new List<float>(actions.GetRow(i)), new List<float>(vars.GetRow(i)));
                else
                    action = agentDecision.Decide(info.stackedVectorObservation, info.visualObservations, new List<float>(actions.GetRow(i)));
                var tempAction = new TakeActionOutput();
                tempAction.outputAction = action;
                result[agent] = tempAction;
                if(BrainToTrain.brainParameters.vectorActionSpaceType == SpaceType.continuous)
                    actionDiff += Enumerable.Zip(action, actions.GetRow(i), (a, b) => Mathf.Sqrt((a - b) * (a - b))).Aggregate((a, v) => a + v);
                else
                    actionDiff += Enumerable.Zip(action, actions.GetRow(i), (a, b) => (Mathf.RoundToInt(a) == Mathf.RoundToInt(b))?0:1).Aggregate((a, v) => a + v);
                agentNumWithDecision++;
            }
            else
            {
                //use result from neural network directly
                var tempAction = new TakeActionOutput();
                tempAction.outputAction = actions.GetRow(i);
                result[agent] = tempAction;
            }
            i++;
        }

        if (agentNumWithDecision > 0)
        {
            stats.AddData("action difference", actionDiff/ agentNumWithDecision);
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
        for (int i = 0; i < parametersMimic.numIterationPerTrain; ++i)
        {
            var samples = dataBuffer.RandomSample(parametersMimic.batchSize, fetches.ToArray());

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

            float temoLoss = modelSL.TrainBatch(vectorObservations, visualObservations, actions);
            loss += temoLoss;
        }

        stats.AddData("loss", loss / parametersMimic.numIterationPerTrain);
    }





    public void SaveTrainingData()
    {
        if (string.IsNullOrEmpty(trainingDataSaveFileName))
        {
            Debug.Log("trainingDataSaveFileName empty. No training data saved.");
            return;
        }
        var binFormatter = new BinaryFormatter();
        var mStream = new MemoryStream();

        binFormatter.Serialize(mStream, dataBuffer);
        var data = mStream.ToArray();
        
        string fullPath = Path.GetFullPath(Path.Combine(checkpointPath, trainingDataSaveFileName));
        fullPath = fullPath.Replace('/', Path.DirectorySeparatorChar);
        fullPath = fullPath.Replace('\\', Path.DirectorySeparatorChar);

        File.WriteAllBytes(fullPath, data);
        Debug.Log("Saved training data to " + fullPath);


    }
    public void LoadTrainingData()
    {
        if (string.IsNullOrEmpty(trainingDataSaveFileName))
        {
            Debug.Log("trainingDataSaveFileName empty. No training data loaded.");
            return;
        }
        string fullPath = Path.GetFullPath(Path.Combine(checkpointPath, trainingDataSaveFileName));

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

        object deserialized = binFormatter.Deserialize(mStream);

        if (deserialized is DataBuffer)
        {
            dataBuffer = (DataBuffer)binFormatter.Deserialize(mStream);
            Debug.Log("Loaded training data from " + fullPath);
        }else if(deserialized is SortedRawHistory)
        {
            dataBuffer = ((SortedRawHistory)deserialized).AddToDataBuffer(BrainToTrain.brainParameters);
        }
        else
        {
            Debug.LogError("Training data format not supported");
        }
    }

}
