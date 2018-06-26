using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Accord.Math;
using System.Linq;
using System.Runtime.InteropServices;
using System.IO;

public class TrainerPPO : Trainer
{

    public RLModelPPO modelRef;
    public TrainerParamsPPO parameters;
    public Brain BrainToTrain { get; private set; }
    
    protected DataBuffer dataBuffer;
    public int DataCountStored { get { return dataBuffer.CurrentCount; } }

    public int Steps { get; protected set; } = 0;

    protected Dictionary<Agent, List<float>> statesEpisodeHistory = null;
    protected Dictionary<Agent, List<float>> rewardsEpisodeHistory = null;
    protected Dictionary<Agent, List<float>> actionsEpisodeHistory = null;
    protected Dictionary<Agent, List<float>> actionprobsEpisodeHistory = null;
    protected Dictionary<Agent, List<float>> valuesEpisodeHistory = null;
    protected Dictionary<Agent, List<List<float[,,]>>> visualEpisodeHistory = null;

    StatsLogger stats;
    protected Dictionary<Agent, float> accumulatedRewards;
    protected Dictionary<Agent, int> episodeSteps;

    public bool continueFromCheckpoint = true;
    public string checkpointPath = @"Assets\testcheckpoint.bytes";

    public override void SetBrain(Brain brain)
    {
        this.BrainToTrain = brain;
    }
    public override void Initialize()
    {
        //initialize all data buffers
        statesEpisodeHistory = new Dictionary<Agent, List<float>>();
        rewardsEpisodeHistory = new Dictionary<Agent, List<float>>();
        actionsEpisodeHistory = new Dictionary<Agent, List<float>>();
        actionprobsEpisodeHistory = new Dictionary<Agent, List<float>>();
        valuesEpisodeHistory = new Dictionary<Agent, List<float>>();
        visualEpisodeHistory = new Dictionary<Agent, List<List<float[,,]>>>();

        accumulatedRewards = new Dictionary<Agent, float>();
        episodeSteps = new Dictionary<Agent, int>();


        var brainParameters = BrainToTrain.brainParameters;

        List<DataBuffer.DataInfo> allBufferData = new List<DataBuffer.DataInfo>() {
            new DataBuffer.DataInfo("Action", typeof(float), new int[] { brainParameters.vectorActionSpaceType == SpaceType.continuous ? brainParameters.vectorActionSize : 1 }),
            new DataBuffer.DataInfo("ActionProb", typeof(float), new int[] { brainParameters.vectorActionSpaceType == SpaceType.continuous ? brainParameters.vectorActionSize : 1 }),
            new DataBuffer.DataInfo("TargetValue", typeof(float), new int[] { 1 }),
            new DataBuffer.DataInfo("Advantage", typeof(float), new int[] { 1 })
        };

        if (brainParameters.vectorObservationSize > 0)
            allBufferData.Add(new DataBuffer.DataInfo("VectorObservation", typeof(float), new int[] { brainParameters.vectorObservationSpaceType == SpaceType.continuous ? brainParameters.vectorObservationSize * brainParameters.numStackedVectorObservations : 1 }));

        for(int i = 0; i < brainParameters.cameraResolutions.Length; ++i)
        {
            int width = brainParameters.cameraResolutions[i].width;
            int height = brainParameters.cameraResolutions[i].height;
            int channels;
            if (brainParameters.cameraResolutions[i].blackAndWhite)
                channels = 1;
            else
                channels = 3;

            allBufferData.Add(new DataBuffer.DataInfo("VisualObservation"+i, typeof(float), new int[] { height , height, channels }));
        }
        dataBuffer = new DataBuffer(parameters.bufferSizeForTrain * 2, allBufferData.ToArray());

        //initialize loggers and neuralnetowrk model
        stats = new StatsLogger();
        modelRef.Initialize(BrainToTrain,parameters);

        if (continueFromCheckpoint)
        {
            Load(); 
        }
    }

    public override void Update()
    {
        modelRef.SetLearningRate(parameters.learningRate);
        base.Update();
    }


    public override void AddExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo, TakeActionOutput actionOutput)
    {
        var agentList = currentInfo.Keys;
        foreach (var agent in agentList)
        {
            if (!statesEpisodeHistory.ContainsKey(agent))
            {
                statesEpisodeHistory[agent] = new List<float>();
                rewardsEpisodeHistory[agent] = new List<float>();
                actionsEpisodeHistory[agent] = new List<float>();
                actionprobsEpisodeHistory[agent] = new List<float>();
                valuesEpisodeHistory[agent] = new List<float>();
                visualEpisodeHistory[agent] = new List<List<float[,,]>>();
                foreach(var c in currentInfo[agent].visualObservations)
                {
                    visualEpisodeHistory[agent].Add(new List<float[,,]>());
                }
                accumulatedRewards[agent] = 0;
            }
            if(currentInfo[agent].stackedVectorObservation.Count > 0)
                statesEpisodeHistory[agent].AddRange(currentInfo[agent].stackedVectorObservation.ToArray());
            rewardsEpisodeHistory[agent].Add(newInfo[agent].reward);
            actionsEpisodeHistory[agent].AddRange(actionOutput.outputAction[agent]);
            actionprobsEpisodeHistory[agent].AddRange(actionOutput.allProbabilities[agent]);
            valuesEpisodeHistory[agent].Add(actionOutput.value[agent]);

            //add the visual observations
            for(int i = 0; i < BrainToTrain.brainParameters.cameraResolutions.Length; ++i)
            {
                var res = BrainToTrain.brainParameters.cameraResolutions[i];
                visualEpisodeHistory[agent][i].Add(TextureToArray(currentInfo[agent].visualObservations[i], res.blackAndWhite));
            }

            accumulatedRewards[agent] += newInfo[agent].reward;
            if(agent.GetStepCount() != 0)
                episodeSteps[agent] = agent.GetStepCount();
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
        if(Steps% parameters.saveModelInterval == 0)
        {
            Save();
        }
    }

    public override bool IsReadyUpdate()
    {
        return dataBuffer.CurrentCount >= parameters.bufferSizeForTrain;
    }

    public override void ProcessExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo)
    {
        var agentList = currentInfo.Keys;
        foreach (var agent in agentList)
        {
            var agentNewInfo = newInfo[agent];
            if (agentNewInfo.done || agentNewInfo.maxStepReached)
            {
                //update process the episode data for PPO.
                float nextValue = modelRef.EvaluateValue(Matrix.Reshape(agentNewInfo.stackedVectorObservation.ToArray(),1, agentNewInfo.stackedVectorObservation.Count),
                    CreateVisualIInputBatch(newInfo, new List<Agent>() { agent },BrainToTrain.brainParameters.cameraResolutions))[0];
                var advantages = RLUtils.GeneralAdvantageEst(rewardsEpisodeHistory[agent].ToArray(),
                    valuesEpisodeHistory[agent].ToArray(), parameters.rewardDiscountFactor, parameters.rewardGAEFactor, nextValue);
                float[] targetValues = new float[advantages.Length];

                var valueHistory = valuesEpisodeHistory[agent];
                for (int i = 0; i < targetValues.Length; ++i)
                {
                    targetValues[i] = advantages[i] + valueHistory[i];
                }
                
                //add those processed data to the buffer

                List<ValueTuple<string, Array>> dataToAdd = new List<ValueTuple<string, Array>>();
                dataToAdd.Add(ValueTuple.Create<string, Array>("Action", actionsEpisodeHistory[agent].ToArray()));
                dataToAdd.Add(ValueTuple.Create<string, Array>("ActionProb", actionprobsEpisodeHistory[agent].ToArray()));
                dataToAdd.Add(ValueTuple.Create<string, Array>("TargetValue", targetValues));
                dataToAdd.Add(ValueTuple.Create<string, Array>("Advantage", advantages));
                if (statesEpisodeHistory[agent].Count > 0)
                    dataToAdd.Add(ValueTuple.Create<string, Array>("VectorObservation", statesEpisodeHistory[agent].ToArray()));

               for(int i = 0; i < visualEpisodeHistory[agent].Count; ++i)
                {
                    dataToAdd.Add(ValueTuple.Create<string, Array>("VisualObservation" + i, DataBuffer.ListToArray(visualEpisodeHistory[agent][i])));
                    visualEpisodeHistory[agent][i].Clear();
                }
                //clear the temperary data record
                statesEpisodeHistory[agent].Clear();
                rewardsEpisodeHistory[agent].Clear();
                actionsEpisodeHistory[agent].Clear();
                actionprobsEpisodeHistory[agent].Clear();
                valuesEpisodeHistory[agent].Clear();

                dataBuffer.AddData(dataToAdd.ToArray());
                
                //update stats
                stats.AddData("accumulatedRewards",accumulatedRewards[agent],parameters.rewardLogInterval);
                accumulatedRewards[agent] = 0;
                stats.AddData("episodeSteps", episodeSteps[agent], parameters.rewardLogInterval);
                episodeSteps[agent] = 0;
                
            }
        }
    }

    public override TakeActionOutput TakeAction(Dictionary<Agent, AgentInfo> agentInfos)
    {
        var result = new TakeActionOutput();
        result.allProbabilities = new Dictionary<Agent, float[]>();
        result.outputAction = new Dictionary<Agent, float[]>();
        result.value = new Dictionary<Agent, float>();

        var agentList = new List<Agent>(agentInfos.Keys);

        float[,] vectorObsAll = CreateVectorIInputBatch(agentInfos, agentList);
        var visualObsAll = CreateVisualIInputBatch(agentInfos, agentList, BrainToTrain.brainParameters.cameraResolutions);


        float[,] actionProbs = null;
        var actions = modelRef.EvaluateAction(vectorObsAll, out actionProbs, visualObsAll);
        var values = modelRef.EvaluateValue(vectorObsAll, visualObsAll);


        int i = 0;
        foreach (var agent in agentList)
        {
            result.allProbabilities[agent] = actionProbs.GetRow(i);
            result.outputAction[agent] = actions.GetRow(i);
            result.value[agent] = values[i];
            i++;
        }
        

        return result;
    }



    public override void UpdateModel()
    {
        var fetches = new List<ValueTuple<string, int, string>>();
        if(BrainToTrain.brainParameters.vectorObservationSize > 0)
            fetches.Add(new ValueTuple<string, int, string>("VectorObservation", 0, "VectorObservation"));
        fetches.Add(new ValueTuple<string, int, string>("Action", 0, "Action"));
        fetches.Add(new ValueTuple<string, int, string>("ActionProb", 0, "ActionProb"));
        fetches.Add(new ValueTuple<string, int, string>("TargetValue", 0, "TargetValue"));
        fetches.Add(new ValueTuple<string, int, string>("Advantage", 0, "Advantage"));
        for(int i = 0; i < BrainToTrain.brainParameters.cameraResolutions.Length; ++i)
        {
            fetches.Add(new ValueTuple<string, int, string>("VisualObservation" + i, 0, "VisualObservation" + i));
        }


        float loss = 0, policyLoss = 0, valueLoss = 0;
        for (int i = 0; i < parameters.numEpochPerTrain; ++i)
        {
            var samples = dataBuffer.SampleBatchesReordered(parameters.batchSize, fetches.ToArray());

            var vectorObsArray = samples.TryGetOr("VectorObservation", null);
            float[,] vectorObservations = vectorObsArray== null? null: vectorObsArray as float[,];
            float[,] actions = (float[,])samples["Action"];
            float[,] actionProbs = (float[,])samples["ActionProb"];
            float[,] targetValues = (float[,])samples["TargetValue"];
            float[,] advantages = (float[,])samples["Advantage"];

            
            List<float[,,,]> visualObservations = null;
            for (int j = 0; j < BrainToTrain.brainParameters.cameraResolutions.Length; ++j)
            {
                if (j == 0)
                    visualObservations = new List<float[,,,]>();
                visualObservations.Add((float[,,,])samples["VisualObservation" + j]);
            }

            int batchCount = targetValues.Length / parameters.batchSize;
            int actionUnitSize = (BrainToTrain.brainParameters.vectorActionSpaceType == SpaceType.continuous ? BrainToTrain.brainParameters.vectorActionSize : 1);
            int totalStateSize = BrainToTrain.brainParameters.vectorObservationSize * BrainToTrain.brainParameters.numStackedVectorObservations;

            float tempLoss = 0, tempPolicyLoss = 0, tempValueLoss = 0;

            for (int j = 0; j < batchCount; ++j)
            {
                
                float[] losses = modelRef.TrainBatch(SubRows(vectorObservations, j * parameters.batchSize , parameters.batchSize ),
                    SubRows(visualObservations, j * parameters.batchSize, parameters.batchSize),
                    SubRows(actions, j * parameters.batchSize , parameters.batchSize ),
                    SubRows(actionProbs, j * parameters.batchSize , parameters.batchSize ),
                    SubRows(targetValues, j * parameters.batchSize, parameters.batchSize).Flatten(),
                    SubRows(advantages, j * parameters.batchSize, parameters.batchSize).Flatten()
                    );
                tempLoss += losses[0];
                tempPolicyLoss += losses[1];
                tempValueLoss += losses[2];
            }

            loss += tempLoss / batchCount; policyLoss += tempPolicyLoss / batchCount; valueLoss += tempValueLoss / batchCount;
        }
        
        stats.AddData("loss", loss / parameters.numEpochPerTrain,parameters.lossLogInterval);
        stats.AddData("policyLoss", policyLoss / parameters.numEpochPerTrain, parameters.lossLogInterval);
        stats.AddData("valueLoss", valueLoss / parameters.numEpochPerTrain, parameters.lossLogInterval);

        dataBuffer.ClearData();
    }


    private static T[,] SubRows<T>(T[,] data, int startRow, int rowCount)
    {
        if (data == null)
            return null;
        int rowLength = data.GetLength(1);
        T[,] result = new T[rowCount, rowLength];
        int typeSize = Marshal.SizeOf(typeof(T));
        Buffer.BlockCopy(data, startRow * rowLength * typeSize, result, 0, rowCount * rowLength * typeSize);
        //Array.Copy(data, index, result, 0, length);
        return result;
    }

    private static List<T[,,,]> SubRows<T>(List<T[,,,]> data, int startRow, int rowCount)
    {
        if (data == null || data.Count == 0)
            return null;
        List<T[,,,]> result = new List<T[,,,]>();
        for(int i = 0; i < data.Count; ++i)
        {
            int rowLength1 = data[i].GetLength(1);
            int rowLength2 = data[i].GetLength(2);
            int rowLength3 = data[i].GetLength(3);
            int rowLengthTotal = rowLength1 * rowLength2 * rowLength3;

            result.Add( new T[rowCount, rowLength1, rowLength2, rowLength3]);
            int typeSize = Marshal.SizeOf(typeof(T));

            Buffer.BlockCopy(data[i], startRow * rowLengthTotal * typeSize, result[i], 0, rowCount * rowLengthTotal * typeSize);
        }

        return result;
    }

    public void Save()
    {
        var data = modelRef.SaveCheckpoint();
        File.WriteAllBytes(checkpointPath, data);
        Debug.Log("Saved Checkpoint to " + Path.Combine(Directory.GetCurrentDirectory(), checkpointPath));
    }
    public void Load()
    {
        string fullPath = Path.Combine(Directory.GetCurrentDirectory(), checkpointPath);
        if (!File.Exists(fullPath))
        {
            Debug.Log("Checkpoint Not exist at: " + Path.Combine(Directory.GetCurrentDirectory(), checkpointPath));
            return;
        }
        var bytes = File.ReadAllBytes(Path.Combine(Directory.GetCurrentDirectory(), checkpointPath));
        modelRef.RestoreCheckpoint(bytes);
        Debug.Log("Loaded from Checkpoint " + Path.Combine(Directory.GetCurrentDirectory(), checkpointPath));
    }

}
