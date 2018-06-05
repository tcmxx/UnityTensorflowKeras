using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class TrainingStats
{
    public List<float> cumulativeReward = new List<float>();
    public List<float> episodeLength = new List<float>();
    public List<float> value = new List<float>();
    public List<float> valueLoss = new List<float>();
    public List<float> policyLoss = new List<float>();
}


public class TrainerPPO : Trainer
{

    public ModelPPO model;
    public TrainerParamsPPO parameters;
    public Brain brain;

    protected DataBuffer dataBuffer;
    public int DataCountStored { get { return dataBuffer.CurrentCount; } }

    public int Steps { get; protected set; } = 0;

    protected Dictionary<Agent, List<float>> statesEpisodeHistory;
    protected Dictionary<Agent, List<float>> rewardsEpisodeHistory;
    protected Dictionary<Agent, List<float>> actionsEpisodeHistory;
    protected Dictionary<Agent, List<float>> actionprobsEpisodeHistory;
    protected Dictionary<Agent, List<float>> valuesEpisodeHistory;


    //TrainingStats stats;


    private void Awake()
    {
        statesEpisodeHistory = new Dictionary<Agent, List<float>>();
        rewardsEpisodeHistory = new Dictionary<Agent, List<float>>();
        actionsEpisodeHistory = new Dictionary<Agent, List<float>>();
        valuesEpisodeHistory = new Dictionary<Agent, List<float>>();


        var brainParameters = brain.brainParameters;

        dataBuffer = new DataBuffer(parameters.bufferSizeForTrain * 2,
            new DataBuffer.DataInfo("State", DataBuffer.DataType.Float, brainParameters.vectorObservationSpaceType == SpaceType.continuous ? brainParameters.vectorObservationSize : 1),
            new DataBuffer.DataInfo("Action", DataBuffer.DataType.Float, brainParameters.vectorActionSpaceType == SpaceType.continuous ? brainParameters.vectorActionSize : 1),
            new DataBuffer.DataInfo("ActionProb", DataBuffer.DataType.Float, brainParameters.vectorActionSpaceType == SpaceType.continuous ? brainParameters.vectorActionSize : 1),
            new DataBuffer.DataInfo("TargetValue", DataBuffer.DataType.Float, 1),
            new DataBuffer.DataInfo("Advantage", DataBuffer.DataType.Float, 1)
            );


        //stats = new TrainingStats();
    }




    public override void AddExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo, TakeActionOutput actionOutput)
    {
        var agentList = currentInfo.Keys;
        foreach (var agent in agentList)
        {
            statesEpisodeHistory[agent].AddRange(currentInfo[agent].vectorObservation.ToArray());
            rewardsEpisodeHistory[agent].Add(newInfo[agent].reward);
            actionsEpisodeHistory[agent].AddRange(actionOutput.outputAction[agent]);
            actionprobsEpisodeHistory[agent].AddRange(actionOutput.allProbabilities[agent]);
            valuesEpisodeHistory[agent].Add(actionOutput.value[agent]);
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

                float nextValue = model.EvaluateValue(agentNewInfo.vectorObservation.ToArray())[0];
                var advantages = RLUtils.GeneralAdvantageEst(rewardsEpisodeHistory[agent].ToArray(),
                    valuesEpisodeHistory[agent].ToArray(), parameters.rewardDiscountFactor, parameters.rewardGAEFactor, nextValue);
                float[] targetValues = new float[advantages.Length];

                var valueHistory = valuesEpisodeHistory[agent];
                for (int i = 0; i < targetValues.Length; ++i)
                {
                    targetValues[i] = advantages[i] + valueHistory[i];
                }
                //test

                dataBuffer.AddData(ValueTuple.Create<string, Array>("State", statesEpisodeHistory[agent].ToArray()),
                    ValueTuple.Create<string, Array>("Action", actionsEpisodeHistory[agent].ToArray()),
                    ValueTuple.Create<string, Array>("ActionProb", actionprobsEpisodeHistory[agent].ToArray()),
                    ValueTuple.Create<string, Array>("TargetValue", targetValues),
                    ValueTuple.Create<string, Array>("Advantage", advantages)
                    );

                statesEpisodeHistory[agent].Clear();
                rewardsEpisodeHistory[agent].Clear();
                actionsEpisodeHistory[agent].Clear();
                actionprobsEpisodeHistory[agent].Clear();
                valuesEpisodeHistory[agent].Clear();
            }
        }
    }

    public override TakeActionOutput TakeAction(Dictionary<Agent, AgentInfo> agentInfos)
    {
        var result = new TakeActionOutput();
        result.allProbabilities = new Dictionary<Agent, float[]>();
        //result.entropy = new Dictionary<Agent, float>();
        result.outputAction = new Dictionary<Agent, float[]>();
        result.value = new Dictionary<Agent, float>();
        //result.memory = new Dictionary<Agent, float[]>();
        //result.textAction = new Dictionary<Agent, string>();

        var agentList = agentInfos.Keys;

        foreach (var agent in agentList)
        {
            //get the action from model
            float[] states = agentInfos[agent].vectorObservation.ToArray();
            float[] actionProbs = null;
            float[] tempAction = model.EvaluateAction(states, out actionProbs, brain.brainParameters.vectorActionSpaceType);
            result.allProbabilities[agent] = actionProbs;
            result.outputAction[agent] = tempAction;

            //get the expected value from model
            float[] value = model.EvaluateValue(states);
            result.value[agent] = value[0];
        }
        

        return result;
    }

    public override void UpdateLastReward()
    {
        //throw new System.NotImplementedException();

    }

    public override void UpdateModel()
    {
        var fetches = new List<ValueTuple<string, int, string>>();
        fetches.Add(new ValueTuple<string, int, string>("State", 0, "State"));
        fetches.Add(new ValueTuple<string, int, string>("Action", 0, "Action"));
        fetches.Add(new ValueTuple<string, int, string>("ActionProb", 0, "ActionProb"));
        fetches.Add(new ValueTuple<string, int, string>("TargetValue", 0, "TargetValue"));
        fetches.Add(new ValueTuple<string, int, string>("Advantage", 0, "Advantage"));

        for (int i = 0; i < parameters.numEpochPerTrain; ++i)
        {
            var samples = dataBuffer.SampleBatchesReordered(parameters.batchSize, fetches.ToArray());
            float[] states = (float[])samples["State"];
            float[] actions = (float[])samples["Action"];
            float[] actionProbs = (float[])samples["ActionProb"];
            float[] targetValues = (float[])samples["TargetValue"];
            float[] advantages = (float[])samples["Advantage"];

            int batchCount = targetValues.Length / parameters.batchSize;
            int actionUnitSize = (brain.brainParameters.vectorActionSpaceType == SpaceType.continuous ? brain.brainParameters.vectorActionSize : 1);
            for (int j = 0; j < batchCount; ++j)
            {
                model.TrainBatch(SubArray(states, j * parameters.batchSize * brain.brainParameters.vectorObservationSize, parameters.batchSize * brain.brainParameters.vectorObservationSize),
                    SubArray(actions, j * parameters.batchSize * actionUnitSize, parameters.batchSize * actionUnitSize),
                    SubArray(actionProbs, j * parameters.batchSize * actionUnitSize, parameters.batchSize * actionUnitSize),
                    SubArray(targetValues, j * parameters.batchSize, parameters.batchSize),
                    SubArray(advantages, j * parameters.batchSize, parameters.batchSize));
            }
        }

        dataBuffer.ClearData();
    }

    public override void WriteSummary()
    {
        //throw new System.NotImplementedException();
    }




    private static T[] SubArray<T>(T[] data, int index, int length)
    {
        T[] result = new T[length];
        Array.Copy(data, index, result, 0, length);
        return result;
    }
}
