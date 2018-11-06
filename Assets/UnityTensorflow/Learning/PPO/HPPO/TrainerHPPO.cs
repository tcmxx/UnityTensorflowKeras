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

public class TrainerHPPO : Trainer
{
    [ShowAllPropertyAttr]
    protected TrainerParamsPPO parametersPPO;

    protected DataBuffer dataBuffer;
    protected DataBuffer policyTrainBuffer;
    protected DataBuffer tempGoodTrainBuffer;

    protected SortedRawHistory policyTrainEpisodeHistory;
    protected SortedRawHistory goodEpisodeHistory;
    public int goodHistoryCount = 20;
    public int goodHistoryRepeat = 1;
    public bool trainPolicy = true;
    public TextAsset exampleEpisodes = null;
    //protected HPPORawHistory exampleHistory;
    public string exampleSaveNameTemp;
    //public int goodHistoryTrainCount = 5;

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
    

    //protected bool nextTrainingPolicy = false;

    //casted modelRef from the base class for convenience
    protected IRLModelHPPO iModelHPPO;

    public override void Initialize()
    {
        iModelHPPO = modelRef as IRLModelHPPO;
        Debug.Assert(iModelHPPO != null, "Please assign a model that implement interface IRLModelHPPO to modelRef");
        parametersPPO = parameters as TrainerParamsPPO;
        Debug.Assert(parametersPPO != null, "Please Specify PPO Trainer Parameters");
        Debug.Assert(BrainToTrain != null, "brain can not be null");


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
        Debug.Assert(brainParameters.vectorActionSize.Length > 0, "Action size can not be zero. Please set it in the brain");
        List<DataBuffer.DataInfo> allBufferData = new List<DataBuffer.DataInfo>() {
            new DataBuffer.DataInfo("Action", typeof(float), new int[] { brainParameters.vectorActionSpaceType == SpaceType.continuous ? brainParameters.vectorActionSize[0] : brainParameters.vectorActionSize.Length }),
            new DataBuffer.DataInfo("ActionProb", typeof(float), new int[] { brainParameters.vectorActionSpaceType == SpaceType.continuous ? brainParameters.vectorActionSize[0] : brainParameters.vectorActionSize.Length }),
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
        policyTrainBuffer = new DataBuffer(allBufferData.ToArray());
        tempGoodTrainBuffer = new DataBuffer(allBufferData.ToArray());
        policyTrainEpisodeHistory = new SortedRawHistory();
        goodEpisodeHistory = new SortedRawHistory(goodHistoryCount);
        //exampleHistory = null;

        if(exampleEpisodes != null)
        {
            goodEpisodeHistory = LoadHistory(exampleEpisodes.bytes);
        }
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
        iModelHPPO.ValueLossWeight = parametersPPO.valueLossWeight;
        iModelHPPO.EntropyLossWeight = parametersPPO.entropyLossWeight;
        iModelHPPO.ClipEpsilon = parametersPPO.clipEpsilon;
        iModelHPPO.ClipValueLoss = parametersPPO.clipValueLoss;
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

        goodEpisodeHistory.Clear();
        policyTrainEpisodeHistory.Clear();


        if (exampleEpisodes != null)
        {
            goodEpisodeHistory = LoadHistory(exampleEpisodes.bytes);
        }
    }

    public override void AddExperience(Dictionary<Agent, AgentInfoInternal> currentInfo, Dictionary<Agent, AgentInfoInternal> newInfo, Dictionary<Agent, TakeActionOutput> actionOutput)
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
                actionMasksEpisodeHistory[agent] = new List<List<float>>();
                if (BrainToTrain.brainParameters.vectorActionSpaceType == SpaceType.discrete)
                {
                    for (int i = 0; i < BrainToTrain.brainParameters.vectorActionSize.Length; ++i)
                    {
                        actionMasksEpisodeHistory[agent].Add(new List<float>());
                    }
                }
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
                visualEpisodeHistory[agent][i].Add(currentInfo[agent].visualObservations[i]);
            }

            if (BrainToTrain.brainParameters.vectorActionSpaceType == SpaceType.discrete)
            {
                int startIndex = 0;
                for (int i = 0; i < BrainToTrain.brainParameters.vectorActionSize.Length; ++i)
                {
                    float[] tempMask = null;
                    if (currentInfo[agent].actionMasks == null)
                    {

                        tempMask = new float[BrainToTrain.brainParameters.vectorActionSize[i]];
                        tempMask.Set(1, 0, tempMask.Length);
                        actionMasksEpisodeHistory[agent][i].AddRange(tempMask);
                    }
                    else
                    {
                        actionMasksEpisodeHistory[agent][i].AddRange(currentInfo[agent].actionMasks.Get(startIndex, startIndex + BrainToTrain.brainParameters.vectorActionSize[i]).Select(x => x ? 0.0f : 1.0f));
                    }
                    startIndex += BrainToTrain.brainParameters.vectorActionSize[i];
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

    public override void ProcessExperience(Dictionary<Agent, AgentInfoInternal> currentInfo, Dictionary<Agent, AgentInfoInternal> newInfo)
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
                    nextValue = iModelHPPO.EvaluateValue(Matrix.Reshape(agentNewInfo.stackedVectorObservation.ToArray(), 1, agentNewInfo.stackedVectorObservation.Count),
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

                policyTrainEpisodeHistory.AddEpisode(statesEpisodeHistory[agent], rewardsEpisodeHistory[agent], actionsEpisodeHistory[agent], visualEpisodeHistory[agent],
                    actionMasksEpisodeHistory[agent],
                    agentNewInfo.stackedVectorObservation, agentNewInfo.visualObservations, agentNewInfo.done && !agentNewInfo.maxStepReached);
                goodEpisodeHistory.AddEpisode(statesEpisodeHistory[agent], rewardsEpisodeHistory[agent], actionsEpisodeHistory[agent], visualEpisodeHistory[agent],
                    actionMasksEpisodeHistory[agent],
                    agentNewInfo.stackedVectorObservation, agentNewInfo.visualObservations, agentNewInfo.done && !agentNewInfo.maxStepReached);

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
                for (int i = 0; i < actionMasksEpisodeHistory[agent].Count; ++i)
                {
                    actionMasksEpisodeHistory[agent][i].Clear();
                }



                //update stats if the agent is not using heuristic
                if (agentNewInfo.done || agentNewInfo.maxStepReached)
                {
                    var agentDecision = agent.GetComponent<AgentDependentDecision>();
                    if (!(isTraining && agentDecision != null && agentDecision.useDecision))// && parametersPPO.useHeuristicChance > 0
                    {
                        stats.AddData("accumulatedRewards", accumulatedRewards[agent]);
                        stats.AddData("episodeSteps", episodeSteps.ContainsKey(agent) ? episodeSteps[agent] : 0);
                    }


                    accumulatedRewards[agent] = 0;
                    episodeSteps[agent] = 0;
                }



            }
        }
    }

    public override Dictionary<Agent, TakeActionOutput> TakeAction(Dictionary<Agent, AgentInfoInternal> agentInfos)
    {
        var result = new Dictionary<Agent, TakeActionOutput>();
        var agentList = new List<Agent>(agentInfos.Keys);
        if (agentList.Count <= 0)
            return result;
        float[,] vectorObsAll = CreateVectorInputBatch(agentInfos, agentList);
        var visualObsAll = CreateVisualInputBatch(agentInfos, agentList, BrainToTrain.brainParameters.cameraResolutions);
        var actionMasks = CreateActionMasks(agentInfos, agentList, BrainToTrain.brainParameters.vectorActionSize);

        float[,] actionProbs = null;
        var values = iModelHPPO.EvaluateValue(vectorObsAll, visualObsAll);
        var actions = iModelHPPO.EvaluateAction(vectorObsAll, out actionProbs, visualObsAll, actionMasks);



        int i = 0;
        foreach (var agent in agentList)
        {
            var agentDecision = agent.GetComponent<AgentDependentDecision>();

            if (isTraining && agentDecision != null && agentDecision.useDecision)// && UnityEngine.Random.Range(0, 1.0f) <= parametersPPO.useHeuristicChance
            {
                //if this agent will use the decision, use it
                var info = agentInfos[agent];
                var action = agentDecision.Decide(info.stackedVectorObservation, info.visualObservations, new List<float>(actions.GetRow(i)));
                float[,] vectorOb = CreateVectorInputBatch(agentInfos, new List<Agent>() { agent });
                var visualOb = CreateVisualInputBatch(agentInfos, new List<Agent>() { agent }, BrainToTrain.brainParameters.cameraResolutions);
                var mask = CreateActionMasks(agentInfos, new List<Agent> { agent }, BrainToTrain.brainParameters.vectorActionSize);
                var probs = iModelHPPO.EvaluateProbability(vectorOb, action.Reshape(1, action.Length), visualOb, mask);

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


        for (int pass = 0; pass < 2; pass++)
        {
            if (pass == 1)
            {
                if (!trainPolicy)
                    break;
                policyTrainBuffer.ClearData();
                policyTrainEpisodeHistory.EvaluateAndAddToDatabuffer(this, policyTrainBuffer);

                tempGoodTrainBuffer.ClearData();
                goodEpisodeHistory.EvaluateAndAddToDatabuffer(this, tempGoodTrainBuffer);
                for (int r = 0; r < goodHistoryRepeat; ++r)
                {
                    policyTrainBuffer.AddData(tempGoodTrainBuffer);
                }
                /*if(exampleHistory != null)
                {
                    tempGoodTrainBuffer.ClearData();
                    exampleHistory.EvaluateAndAddToDatabuffer(this, tempGoodTrainBuffer);
                    policyTrainBuffer.AddData(tempGoodTrainBuffer);
                }*/
            }

            for (int i = 0; i < parametersPPO.numEpochPerTrain; ++i)
            {
                //training from the main data buffer
                Dictionary<string, Array> samples = null;
                if (pass == 0)
                    samples = dataBuffer.SampleBatchesReordered(parametersPPO.batchSize, fetches.ToArray());
                else if (pass == 1)
                {
                    samples = policyTrainBuffer.SampleBatchesReordered(parametersPPO.batchSize, fetches.ToArray());
                }

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
                if (BrainToTrain.brainParameters.vectorActionSpaceType == SpaceType.discrete)
                {
                    actionMasks = new List<float[,]>();
                    for (int b = 0; b < BrainToTrain.brainParameters.vectorActionSize.Length; ++b)
                    {
                        actionMasks.Add((float[,])samples["ActionMask" + b]);
                    }
                }

                //int batchCount = parametersPPO.bufferSizeForTrain / parametersPPO.batchSize;
                int batchCount = advantages.Length / parametersPPO.batchSize;
                //int actionUnitSize = (BrainToTrain.brainParameters.vectorActionSpaceType == SpaceType.continuous ? BrainToTrain.brainParameters.vectorActionSize : 1);
                //int totalStateSize = BrainToTrain.brainParameters.vectorObservationSize * BrainToTrain.brainParameters.numStackedVectorObservations;

                float tempLoss = 0, tempPolicyLoss = 0, tempValueLoss = 0, tempEntropy = 0;
                for (int j = 0; j < batchCount; ++j)
                {
                    int startRow = j * parametersPPO.batchSize;

                    if (pass == 0)
                    {
                        float vLoss = iModelHPPO.TrainValue(vectorObservations.SubRows(startRow, parametersPPO.batchSize),
                            visualObservations.SubRows(startRow, parametersPPO.batchSize),
                            oldValues.SubRows(startRow, parametersPPO.batchSize).Flatten(),
                            targetValues.SubRows(startRow, parametersPPO.batchSize).Flatten())[0];

                        tempValueLoss += vLoss;
                    }
                    else if (pass == 1)
                    {
                        float[] policyLosses = iModelHPPO.TrainPolicy(vectorObservations.SubRows(startRow, parametersPPO.batchSize),
                            visualObservations.SubRows(startRow, parametersPPO.batchSize),
                            actions.SubRows(startRow, parametersPPO.batchSize),
                            advantages.Get(startRow, (j + 1) * parametersPPO.batchSize),
                            actionProbs.SubRows(startRow, parametersPPO.batchSize),
                            actionMasks.SubRows(startRow, parametersPPO.batchSize)
                            );
                        tempLoss += policyLosses[0];
                        tempPolicyLoss += policyLosses[1];
                        tempEntropy += policyLosses[2];
                    }

                }

                loss += tempLoss / (batchCount);
                policyLoss += tempPolicyLoss / (batchCount);
                valueLoss += tempValueLoss / (batchCount);
                entropy += tempEntropy / (batchCount);
            }
        }

        ///-----------train with good memories-----------------
        /*policyTrainBuffer.ClearData();
        goodEpisodeHistory.EvaluateAndAddToDatabuffer(this, policyTrainBuffer);
        for (int i = 0; i < goodHistoryTrainCount; ++i)
        {
            var samples = policyTrainBuffer.RandomSample(parametersPPO.batchSize, fetches.ToArray());
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
            if (BrainToTrain.brainParameters.vectorActionSpaceType == SpaceType.discrete)
            {
                actionMasks = new List<float[,]>();
                for (int b = 0; b < BrainToTrain.brainParameters.vectorActionSize.Length; ++b)
                {
                    actionMasks.Add((float[,])samples["ActionMask" + b]);
                }
            }

            float[] policyLosses = iModelHPPO.TrainPolicy(vectorObservations,visualObservations,actions,advantages,actionProbs,actionMasks);
            //not count the losses here for now
        }*/
        //------------done train with good memory-------------


        stats.AddData("loss", loss / parametersPPO.numEpochPerTrain);
        stats.AddData("policyLoss", policyLoss / parametersPPO.numEpochPerTrain);
        stats.AddData("entropy", entropy / parametersPPO.numEpochPerTrain);

        stats.AddData("valueLoss", valueLoss / parametersPPO.numEpochPerTrain);
        //}
        dataBuffer.ClearData();

        policyTrainEpisodeHistory.Clear();
        SaveHistory(goodEpisodeHistory, exampleSaveNameTemp);
        goodEpisodeHistory.Clear();
        


        //nextTrainingPolicy = !nextTrainingPolicy;
    }



    public void EvaluateEpisode(List<float> vectorObsEpisodeHistory, List<List<float[,,]>> visualEpisodeHistory, List<float> actionsEpisodeHistory, List<float> rewardsEpisodeHistory, List<List<float>> actionMasksEpisodeHistory,
         out float[] values, out float[,] actionProbs, out float[] targetValues, out float[] advantages,
        bool isDone, List<float> finalVectorObs = null, List<float[,,]> finalVisualObs = null)
    {
        int obsSize = BrainToTrain.brainParameters.vectorObservationSize * BrainToTrain.brainParameters.numStackedVectorObservations;
        float[,] vectorObs = vectorObsEpisodeHistory.Reshape(obsSize);
        var visualObs = CreateVisualInputBatch(visualEpisodeHistory, BrainToTrain.brainParameters.cameraResolutions);
        var actionMasks = CreateActionMasks(actionMasksEpisodeHistory, BrainToTrain.brainParameters.vectorActionSize);

        values = iModelHPPO.EvaluateValue(vectorObs, visualObs);
        actionProbs = iModelHPPO.EvaluateProbability(vectorObs,
            actionsEpisodeHistory.Reshape(BrainToTrain.brainParameters.vectorActionSpaceType == SpaceType.continuous ? BrainToTrain.brainParameters.vectorActionSize[0] : BrainToTrain.brainParameters.vectorActionSize.Length),
            visualObs, actionMasks);



        //update process the episode data for PPO.
        float nextValue = 0;
        if (isDone)
        {
            nextValue = 0;  //this is very important!
        }
        else
        {
            List<List<float[,,]>> visualTemp = new List<List<float[,,]>>();
            foreach (var v in finalVisualObs)
            {
                var t = new List<float[,,]>();
                t.Add(v);
                visualTemp.Add(t);
            }
            nextValue = iModelHPPO.EvaluateValue(finalVectorObs.Reshape(obsSize), CreateVisualInputBatch(visualTemp, BrainToTrain.brainParameters.cameraResolutions))[0];
        }

        advantages = RLUtils.GeneralAdvantageEst(rewardsEpisodeHistory.ToArray(), values, parametersPPO.rewardDiscountFactor, parametersPPO.rewardGAEFactor, nextValue);
        targetValues = new float[advantages.Length];
        for (int i = 0; i < targetValues.Length; ++i)
        {
            targetValues[i] = advantages[i] + values[i];
        }

    }
    
    public void SaveHistory(SortedRawHistory history, string savefilename)
    {
        if (string.IsNullOrEmpty(savefilename))
        {
            Debug.Log("savefilename empty. history not saved.");
            return;
        }
        //serailzie the data and save it to path
        var binFormatter = new BinaryFormatter();
        var mStream = new MemoryStream();
        binFormatter.Serialize(mStream, history);
        byte[] data = mStream.ToArray();
        var fullPath = Path.GetFullPath(Path.Combine(checkpointPath, savefilename));
        fullPath = fullPath.Replace('/', Path.DirectorySeparatorChar);
        fullPath = fullPath.Replace('\\', Path.DirectorySeparatorChar);

        Directory.CreateDirectory(Path.GetDirectoryName(fullPath));
        File.WriteAllBytes(fullPath, data);
        Debug.Log("history saved to " + fullPath);
    }

    public SortedRawHistory LoadHistory( byte[] data)
    {
        var mStream = new MemoryStream(data);
        var binFormatter = new BinaryFormatter();
        var deserializedData = binFormatter.Deserialize(mStream);
        if (deserializedData is SortedRawHistory)
        {
            return deserializedData as SortedRawHistory;
        }
        else
            return null;
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


}
