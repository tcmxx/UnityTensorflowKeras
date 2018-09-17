using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Accord.Math;
using System.Linq;
using Accord.Statistics;
using System.Runtime.InteropServices;
using System.IO;
using MLAgents;
using System.Runtime.Serialization.Formatters.Binary;

public class TrainerPPO : Trainer
{
    [ShowAllPropertyAttr]
    protected TrainerParamsPPO parametersPPO;

    protected DataBuffer dataBuffer;
    public int DataCountStored { get { return dataBuffer.CurrentCount; } }

    protected Dictionary<Agent, List<float>> statesEpisodeHistory = null;
    protected Dictionary<Agent, List<float>> rewardsEpisodeHistory = null;
    protected Dictionary<Agent, List<float>> actionsEpisodeHistory = null;
    protected Dictionary<Agent, List<float>> actionprobsEpisodeHistory = null;
    protected Dictionary<Agent, List<float>> valuesEpisodeHistory = null;
    protected Dictionary<Agent, List<List<float[,,]>>> visualEpisodeHistory = null;
    protected Dictionary<Agent, List<List<float>>> actionMasksEpisodeHistory = null;

    public StatsLogger stats { get; protected set; }
    protected Dictionary<Agent, float> accumulatedRewards;
    protected Dictionary<Agent, int> episodeSteps;


    //casted modelRef from the base class for convenience
    protected IRLModelPPO iModelPPO;

    public override void Initialize()
    {
        iModelPPO = modelRef as IRLModelPPO;
        Debug.Assert(iModelPPO != null, "Please assign a model that implement interface IRLModelPPO to modelRef");
        parametersPPO = parameters as TrainerParamsPPO;
        Debug.Assert(parametersPPO != null, "Please Specify PPO Trainer Parameters");



        //initialize all data buffers
        statesEpisodeHistory = new Dictionary<Agent, List<float>>();
        rewardsEpisodeHistory = new Dictionary<Agent, List<float>>();
        actionsEpisodeHistory = new Dictionary<Agent, List<float>>();
        actionprobsEpisodeHistory = new Dictionary<Agent, List<float>>();
        valuesEpisodeHistory = new Dictionary<Agent, List<float>>();
        visualEpisodeHistory = new Dictionary<Agent, List<List<float[,,]>>>();
        actionMasksEpisodeHistory = new Dictionary<Agent, List<List<float>>>();

        accumulatedRewards = new Dictionary<Agent, float>();
        episodeSteps = new Dictionary<Agent, int>();


        var brainParameters = BrainToTrain.brainParameters;
        Debug.Assert(brainParameters.vectorActionSize.Length <= 1, "Action branching is not supported yet");

        List<DataBuffer.DataInfo> allBufferData = new List<DataBuffer.DataInfo>() {
            new DataBuffer.DataInfo("Action", typeof(float), new int[] { brainParameters.vectorActionSpaceType == SpaceType.continuous ? brainParameters.vectorActionSize[0] : 1 }),
            new DataBuffer.DataInfo("ActionProb", typeof(float), new int[] { brainParameters.vectorActionSpaceType == SpaceType.continuous ? brainParameters.vectorActionSize[0] : 1 }),
            new DataBuffer.DataInfo("TargetValue", typeof(float), new int[] { 1 }),
            new DataBuffer.DataInfo("OldValue", typeof(float), new int[] { 1 }),
            new DataBuffer.DataInfo("Advantage", typeof(float), new int[] { 1 })
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

        if (brainParameters.vectorActionSpaceType == SpaceType.discrete)
        {
            for (int i = 0; i < brainParameters.vectorActionSize.Length; ++i)
            {
                allBufferData.Add(new DataBuffer.DataInfo("ActionMask" + i, typeof(float), new int[] { brainParameters.vectorActionSize[i] }));
            }
        }

        dataBuffer = new DataBuffer(allBufferData.ToArray());

        //initialize loggers and neuralnetowrk model
        stats = new StatsLogger();

        modelRef.Initialize(BrainToTrain.brainParameters, isTraining, parameters);
        if (continueFromCheckpoint)
        {
            LoadModel();
        }
    }

    protected override void FixedUpdate()
    {
        iModelPPO.ValueLossWeight = parametersPPO.valueLossWeight;
        iModelPPO.EntropyLossWeight = parametersPPO.entropyLossWeight;
        iModelPPO.ClipEpsilon = parametersPPO.clipEpsilon;
        iModelPPO.ClipValueLoss = parametersPPO.clipValueLoss;
        base.FixedUpdate();
    }

    public override void IncrementStep()
    {
        base.IncrementStep();
        if (GetStep() % parametersPPO.logInterval == 0 && GetStep() != 0)
        {
            stats.LogAllCurrentData(GetStep());
        }
    }
    public override void ResetTrainer()
    {
        base.ResetTrainer();

        var agents = statesEpisodeHistory.Keys;
        stats.ClearAll();
        statesEpisodeHistory.Clear();
        rewardsEpisodeHistory.Clear();
        actionprobsEpisodeHistory.Clear();
        actionsEpisodeHistory.Clear();
        valuesEpisodeHistory.Clear();
        accumulatedRewards.Clear();
        episodeSteps.Clear();
        dataBuffer.ClearData();
        foreach (var agent in agents)
        {
            if (agent)
                agent.AgentReset();
        }
    }

    public override void AddExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo, Dictionary<Agent, TakeActionOutput> actionOutput)
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
                foreach (var c in currentInfo[agent].visualObservations)
                {
                    visualEpisodeHistory[agent].Add(new List<float[,,]>());
                }
                accumulatedRewards[agent] = 0;
            }
            if (currentInfo[agent].stackedVectorObservation.Count > 0)
                statesEpisodeHistory[agent].AddRange(currentInfo[agent].stackedVectorObservation.ToArray());
            rewardsEpisodeHistory[agent].Add(newInfo[agent].reward);
            actionsEpisodeHistory[agent].AddRange(actionOutput[agent].outputAction);
            actionprobsEpisodeHistory[agent].AddRange(actionOutput[agent].allProbabilities);
            valuesEpisodeHistory[agent].Add(actionOutput[agent].value);

            //add the visual observations
            for (int i = 0; i < BrainToTrain.brainParameters.cameraResolutions.Length; ++i)
            {
                var res = BrainToTrain.brainParameters.cameraResolutions[i];
                visualEpisodeHistory[agent][i].Add(TextureToArray(currentInfo[agent].visualObservations[i], res.blackAndWhite));
            }

            if (BrainToTrain.brainParameters.vectorActionSpaceType == SpaceType.discrete)
            {
                int startIndex = 0;
                for (int i = 0; i < BrainToTrain.brainParameters.vectorActionSize.Length; ++i)
                {
                    actionMasksEpisodeHistory[agent][i].AddRange(currentInfo[agent].actionMasks.Get(startIndex, startIndex + BrainToTrain.brainParameters.vectorActionSize[i]).Select(x=>x?0.0f:1.0f));
                }
            }
            accumulatedRewards[agent] += newInfo[agent].reward;
            if (agent.GetStepCount() != 0)
                episodeSteps[agent] = agent.GetStepCount();
        }

    }



    public override bool IsReadyUpdate()
    {
        return dataBuffer.CurrentCount >= parametersPPO.bufferSizeForTrain;
    }

    public override void ProcessExperience(Dictionary<Agent, AgentInfo> currentInfo, Dictionary<Agent, AgentInfo> newInfo)
    {
        var agentList = currentInfo.Keys;
        foreach (var agent in agentList)
        {
            var agentNewInfo = newInfo[agent];
            if (agentNewInfo.done || agentNewInfo.maxStepReached || rewardsEpisodeHistory[agent].Count > parametersPPO.timeHorizon)
            {
                //update process the episode data for PPO.
                float nextValue = 0;

                if (agentNewInfo.done && !agentNewInfo.maxStepReached)
                {
                    nextValue = 0;  //this is very important!
                }
                else
                {
                    nextValue = iModelPPO.EvaluateValue(Matrix.Reshape(agentNewInfo.stackedVectorObservation.ToArray(), 1, agentNewInfo.stackedVectorObservation.Count),
                        CreateVisualInputBatch(newInfo, new List<Agent>() { agent }, BrainToTrain.brainParameters.cameraResolutions))[0];
                }

                var valueHistory = valuesEpisodeHistory[agent].ToArray();
                var advantages = RLUtils.GeneralAdvantageEst(rewardsEpisodeHistory[agent].ToArray(),
                    valueHistory, parametersPPO.rewardDiscountFactor, parametersPPO.rewardGAEFactor, nextValue);
                float[] targetValues = new float[advantages.Length];
                for (int i = 0; i < targetValues.Length; ++i)
                {
                    targetValues[i] = advantages[i] + valueHistory[i];
                }

                //add those processed data to the buffer

                List<ValueTuple<string, Array>> dataToAdd = new List<ValueTuple<string, Array>>();
                dataToAdd.Add(ValueTuple.Create<string, Array>("Action", actionsEpisodeHistory[agent].ToArray()));
                dataToAdd.Add(ValueTuple.Create<string, Array>("ActionProb", actionprobsEpisodeHistory[agent].ToArray()));
                dataToAdd.Add(ValueTuple.Create<string, Array>("TargetValue", targetValues));
                dataToAdd.Add(ValueTuple.Create<string, Array>("OldValue", valueHistory));
                dataToAdd.Add(ValueTuple.Create<string, Array>("Advantage", advantages));
                if (statesEpisodeHistory[agent].Count > 0)
                    dataToAdd.Add(ValueTuple.Create<string, Array>("VectorObservation", statesEpisodeHistory[agent].ToArray()));
                for (int i = 0; i < visualEpisodeHistory[agent].Count; ++i)
                {
                    dataToAdd.Add(ValueTuple.Create<string, Array>("VisualObservation" + i, DataBuffer.ListToArray(visualEpisodeHistory[agent][i])));
                }
                for (int i = 0; i < actionMasksEpisodeHistory[agent].Count; ++i)
                {
                    dataToAdd.Add(ValueTuple.Create<string, Array>("ActionMask" + i, actionMasksEpisodeHistory[agent][i].ToArray()));
                }

                dataBuffer.AddData(dataToAdd.ToArray());

                //clear the temperary data record
                statesEpisodeHistory[agent].Clear();
                rewardsEpisodeHistory[agent].Clear();
                actionsEpisodeHistory[agent].Clear();
                actionprobsEpisodeHistory[agent].Clear();
                valuesEpisodeHistory[agent].Clear();
                for (int i = 0; i < visualEpisodeHistory[agent].Count; ++i)
                {
                    visualEpisodeHistory[agent][i].Clear();
                }




                //update stats if the agent is not using heuristic
                if (agentNewInfo.done || agentNewInfo.maxStepReached)
                {
                    stats.AddData("accumulatedRewards", accumulatedRewards[agent]);
                    stats.AddData("episodeSteps", episodeSteps[agent]);


                    accumulatedRewards[agent] = 0;
                    episodeSteps[agent] = 0;
                }

            }
        }
    }

    public override Dictionary<Agent, TakeActionOutput> TakeAction(Dictionary<Agent, AgentInfo> agentInfos)
    {
        var result = new Dictionary<Agent, TakeActionOutput>();
        var agentList = new List<Agent>(agentInfos.Keys);
        if (agentList.Count <= 0)
            return result;
        float[,] vectorObsAll = CreateVectorInputBatch(agentInfos, agentList);
        var visualObsAll = CreateVisualInputBatch(agentInfos, agentList, BrainToTrain.brainParameters.cameraResolutions);
        var actionMasks = CreateActionMasks(agentInfos, agentList, BrainToTrain.brainParameters.vectorActionSize);

        float[,] actionProbs = null;
        var values = iModelPPO.EvaluateValue(vectorObsAll, visualObsAll);
        var actions = iModelPPO.EvaluateAction(vectorObsAll, out actionProbs, visualObsAll, actionMasks);



        int i = 0;
        foreach (var agent in agentList)
        {
            var agentDecision = agent.GetComponent<AgentDependentDecision>();

            if (isTraining && agentDecision != null && agentDecision.useDecision && UnityEngine.Random.Range(0, 1.0f) <= parametersPPO.useHeuristicChance)
            {
                //if this agent will use the decision, use it
                var info = agentInfos[agent];
                var action = agentDecision.Decide(info.stackedVectorObservation, info.visualObservations, new List<float>(actions.GetRow(i)));
                float[,] vectorOb = CreateVectorInputBatch(agentInfos, new List<Agent>() { agent });
                var visualOb = CreateVisualInputBatch(agentInfos, new List<Agent>() { agent }, BrainToTrain.brainParameters.cameraResolutions);
                var probs = iModelPPO.EvaluateProbability(vectorOb, action.Reshape(1, action.Length), visualOb, actionMasks);

                var temp = new TakeActionOutput();
                temp.allProbabilities = probs.GetRow(0);
                temp.outputAction = action;
                temp.value = values[i];
                result[agent] = temp;
            }
            else
            {
                var temp = new TakeActionOutput();
                temp.allProbabilities = actionProbs.GetRow(i);
                temp.outputAction = actions.GetRow(i);
                temp.value = values[i];
                result[agent] = temp;
            }
            i++;
        }


        return result;
    }



    public override void UpdateModel()
    {
        var fetches = new List<ValueTuple<string, int, string>>();
        if (BrainToTrain.brainParameters.vectorObservationSize > 0)
            fetches.Add(new ValueTuple<string, int, string>("VectorObservation", 0, "VectorObservation"));
        fetches.Add(new ValueTuple<string, int, string>("Action", 0, "Action"));
        fetches.Add(new ValueTuple<string, int, string>("ActionProb", 0, "ActionProb"));
        fetches.Add(new ValueTuple<string, int, string>("TargetValue", 0, "TargetValue"));
        fetches.Add(new ValueTuple<string, int, string>("OldValue", 0, "OldValue"));
        fetches.Add(new ValueTuple<string, int, string>("Advantage", 0, "Advantage"));
        for (int i = 0; i < BrainToTrain.brainParameters.cameraResolutions.Length; ++i)
        {
            fetches.Add(new ValueTuple<string, int, string>("VisualObservation" + i, 0, "VisualObservation" + i));
        }
        if (BrainToTrain.brainParameters.vectorActionSpaceType == SpaceType.discrete)
        {
            for (int i = 0; i < BrainToTrain.brainParameters.vectorActionSize.Length; ++i)
            {
                fetches.Add(new ValueTuple<string, int, string>("ActionMask" + i, 0, "ActionMask" + i));
            }
                
        }

        float loss = 0, policyLoss = 0, valueLoss = 0, entropy = 0;
        for (int i = 0; i < parametersPPO.numEpochPerTrain; ++i)
        {
            //training from the main data buffer
            var samples = dataBuffer.SampleBatchesReordered(parametersPPO.batchSize, fetches.ToArray());

            var vectorObsArray = samples.TryGetOr("VectorObservation", null);
            float[,] vectorObservations = vectorObsArray == null ? null : vectorObsArray as float[,];
            float[,] actions = (float[,])samples["Action"];
            float[,] actionProbs = (float[,])samples["ActionProb"];
            float[,] targetValues = (float[,])samples["TargetValue"];
            float[,] oldValues = (float[,])samples["OldValue"];
            float[] advantages = ((float[,])samples["Advantage"]).Flatten();


            //print("Adv before normalizatoin:" + advantages.Mean());

            float advMean = advantages.Mean();
            double advstd = advantages.StandardDeviation();
            for (int n = 0; n < advantages.Length; ++n)
            {
                advantages[n] = (advantages[n] - advMean) / ((float)advstd + 0.0000000001f);
            }



            List<float[,,,]> visualObservations = null;
            for (int j = 0; j < BrainToTrain.brainParameters.cameraResolutions.Length; ++j)
            {
                if (j == 0)
                    visualObservations = new List<float[,,,]>();
                visualObservations.Add((float[,,,])samples["VisualObservation" + j]);
            }

            List<float[,]> actionMasks = null;
            if(BrainToTrain.brainParameters.vectorActionSpaceType == SpaceType.discrete)
            {
                actionMasks = new List<float[,]>();
                for (int b = 0; b < BrainToTrain.brainParameters.vectorActionSize.Length; ++b)
                {
                    actionMasks.Add((float[,])samples["ActionMask" + i]);
                }
            }

            int batchCount = targetValues.Length / parametersPPO.batchSize;
            //int actionUnitSize = (BrainToTrain.brainParameters.vectorActionSpaceType == SpaceType.continuous ? BrainToTrain.brainParameters.vectorActionSize : 1);
            //int totalStateSize = BrainToTrain.brainParameters.vectorObservationSize * BrainToTrain.brainParameters.numStackedVectorObservations;

            float tempLoss = 0, tempPolicyLoss = 0, tempValueLoss = 0, tempEntropy = 0;
            for (int j = 0; j < batchCount; ++j)
            {
                int startRow = j * parametersPPO.batchSize;

                float[] losses = iModelPPO.TrainBatch(SubRows(vectorObservations, startRow, parametersPPO.batchSize),
                    SubRows(visualObservations, startRow, parametersPPO.batchSize),
                    SubRows(actions, startRow, parametersPPO.batchSize),
                    SubRows(actionProbs, startRow, parametersPPO.batchSize),
                    SubRows(targetValues, startRow, parametersPPO.batchSize).Flatten(),
                    SubRows(oldValues, startRow, parametersPPO.batchSize).Flatten(),
                    advantages.Get(startRow, (j + 1) * parametersPPO.batchSize),
                    SubRows(actionMasks, startRow, parametersPPO.batchSize)
                    );
                /*var testSamples = dataBuffer.RandomSample(parametersPPO.batchSize, fetches.ToArray());
                float[] losses = iModelPPO.TrainBatch((float[,])testSamples["VectorObservation"],null,
                    (float[,])testSamples["Action"],
                    (float[,])testSamples["ActionProb"],
                    ((float[,])testSamples["TargetValue"]).Flatten(),
                    ((float[,])testSamples["OldValue"]).Flatten(),
                    ((float[,])testSamples["Advantage"]).Flatten()
                );*/

                //print("policyLoss:" + losses[2]);
                //print("valueLoss:" + losses[1]);
                //print("actions mean:" + actions.Flatten().Mean());

                tempLoss += losses[0];
                tempValueLoss += losses[1];
                tempPolicyLoss += losses[2];
                tempEntropy += losses[3];
            }

            loss += tempLoss / (batchCount);
            policyLoss += tempPolicyLoss / (batchCount);
            valueLoss += tempValueLoss / (batchCount);
            entropy += tempEntropy / (batchCount);
        }

        //log the stats
        stats.AddData("loss", loss / parametersPPO.numEpochPerTrain);
        stats.AddData("policyLoss", policyLoss / parametersPPO.numEpochPerTrain);
        stats.AddData("valueLoss", valueLoss / parametersPPO.numEpochPerTrain);
        stats.AddData("entropy", entropy / parametersPPO.numEpochPerTrain);
        dataBuffer.ClearData();
    }


    public override float[] PostprocessingAction(float[] rawAction)
    {
        if (BrainToTrain.brainParameters.vectorActionSpaceType == SpaceType.continuous)
        {
            return ClipAndNormalize(rawAction, parametersPPO.finalActionClip, parametersPPO.finalActionDownscale);
        }
        else
        {
            return rawAction;
        }
    }

    private float[] ClipAndNormalize(float[] array, float clip, float divide)
    {
        var result = new float[array.Length];
        for (int i = 0; i < array.Length; ++i)
        {
            result[i] = Mathf.Clamp(array[i], -clip, clip) / divide;
        }
        return result;
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
        for (int i = 0; i < data.Count; ++i)
        {
            int rowLength1 = data[i].GetLength(1);
            int rowLength2 = data[i].GetLength(2);
            int rowLength3 = data[i].GetLength(3);
            int rowLengthTotal = rowLength1 * rowLength2 * rowLength3;

            result.Add(new T[rowCount, rowLength1, rowLength2, rowLength3]);
            int typeSize = Marshal.SizeOf(typeof(T));

            Buffer.BlockCopy(data[i], startRow * rowLengthTotal * typeSize, result[i], 0, rowCount * rowLengthTotal * typeSize);
        }

        return result;
    }

    private static List<T[,]> SubRows<T>(List<T[,]> data, int startRow, int rowCount)
    {
        if (data == null || data.Count == 0)
            return null;
        List<T[,]> result = new List<T[,]>();
        for (int i = 0; i < data.Count; ++i)
        {
            int rowLength1 = data[i].GetLength(1);
            int rowLengthTotal = rowLength1;

            result.Add(new T[rowCount, rowLength1]);
            int typeSize = Marshal.SizeOf(typeof(T));

            Buffer.BlockCopy(data[i], startRow * rowLengthTotal * typeSize, result[i], 0, rowCount * rowLengthTotal * typeSize);
        }

        return result;
    }
}
