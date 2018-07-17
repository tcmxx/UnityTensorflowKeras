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
    protected TrainerParamsMimic parametersMimic;

    
    StatsLogger stats;

    protected DataBuffer dataBuffer;



    protected SupervisedLearningModel modelSL;


    public override void Initialize()
    {
        modelSL = modelRef as SupervisedLearningModel;
        Debug.Assert(modelSL != null, "Please assign a SupervisedLearningModel to modelRef");

        parametersMimic = parameters as TrainerParamsMimic;
        Debug.Assert(parametersMimic != null, "Please Specify PPO Trainer Parameters");
        stats = new StatsLogger();
        modelRef.Initialize(BrainToTrain.brainParameters, isTraining, parameters);

        var brainParameters = BrainToTrain.brainParameters;

        //intialize data buffer
        List<DataBuffer.DataInfo> allBufferData = new List<DataBuffer.DataInfo>() {
            new DataBuffer.DataInfo("Action", typeof(float), new int[] { brainParameters.vectorActionSpaceType == SpaceType.continuous ? brainParameters.vectorActionSize : 1 })
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
        dataBuffer = new DataBuffer(parametersMimic.maxBufferSize, allBufferData.ToArray());

        if (continueFromCheckpoint)
        {
            LoadModel();
            LoadTrainingData();
        }
    }

    public override void AddExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo, Dictionary<Agent,TakeActionOutput> actionOutput)
    {
        var agentList = currentInfo.Keys;
        foreach (var agent in agentList)
        {
            var agentDecision = agent.GetComponent<AgentDependentDecision>();
            //only add the data to databuffer if this agent uses decision
            if (agentDecision != null && agentDecision.useDecision)
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
    }

    public override void IncrementStep()
    {
        base.IncrementStep();
        if (GetStep() % parametersMimic.saveModelInterval == 0)
        {
            SaveTrainingData();
        }
    }



    public override bool IsReadyUpdate()
    {
        return parametersMimic.requiredDataBeforeTraining <= dataBuffer.CurrentCount;
    }

    public override void ProcessExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo)
    {
        return;
    }



    public override Dictionary<Agent,TakeActionOutput> TakeAction(Dictionary<Agent, AgentInfo> agentInfos)
    {
        var result = new Dictionary<Agent, TakeActionOutput>();

        var agentList = new List<Agent>(agentInfos.Keys);

        float[,] vectorObsAll = CreateVectorIInputBatch(agentInfos, agentList);
        var visualObsAll = CreateVisualIInputBatch(agentInfos, agentList, BrainToTrain.brainParameters.cameraResolutions);

        float[,] actions = null;
        actions = modelSL.EvaluateAction(vectorObsAll, visualObsAll);

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
                var action = agentDecision.Decide(agent, info.stackedVectorObservation, info.visualObservations, new List<float>(actions.GetRow(i)));
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
            stats.AddData("action difference", actionDiff/ agentNumWithDecision, parametersMimic.actionDiffLogInterval);
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

        stats.AddData("loss", loss / parametersMimic.numIterationPerTrain, parametersMimic.lossLogInterval);
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
