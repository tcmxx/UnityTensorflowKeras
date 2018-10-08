using MLAgents;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using Accord.Math;
using Accord.Statistics;
using System.Runtime.InteropServices;

public class TrainerPPOCMA : Trainer
{

    [ShowAllPropertyAttr]
    protected TrainerParamsPPO parametersPPO;
    public int varHistroyCount = 5;

    protected DataBuffer dataBuffer;
    protected DataBuffer dataBufferTrainingVariance;

    public int DataCountStored { get { return dataBuffer.CurrentCount; } }

    protected Dictionary<Agent, List<float>> statesEpisodeHistory = null;
    protected Dictionary<Agent, List<float>> rewardsEpisodeHistory = null;
    protected Dictionary<Agent, List<float>> actionsEpisodeHistory = null;
    protected Dictionary<Agent, List<float>> valuesEpisodeHistory = null;
    protected Dictionary<Agent, List<List<float[,,]>>> visualEpisodeHistory = null;

    public StatsLogger stats { get; protected set; }
    protected Dictionary<Agent, float> accumulatedRewards;
    protected Dictionary<Agent, int> episodeSteps;
    public Brain BrainToTrain { get; protected set; }
    //protected List<List<float>> pretrainObservationDataCollect;

    public int pretrainingBatches = 0;
    public int pretrainingBatchSize = 64;

    //casted modelRef from the base class for convenience
    protected IRLModelPPOCMA iModelPPO;

    public override void Initialize(Brain brain)
    {
        iModelPPO = modelRef as IRLModelPPOCMA;
        Debug.Assert(iModelPPO != null, "Please assign a model that implement interface IRLModelPPO to modelRef");
        parametersPPO = parameters as TrainerParamsPPO;
        Debug.Assert(parametersPPO != null, "Please Specify PPO Trainer Parameters");
        BrainToTrain = brain;
        Debug.Assert(BrainToTrain != null, "brain can not be null");
        


        //initialize all data buffers
        statesEpisodeHistory = new Dictionary<Agent, List<float>>();
        rewardsEpisodeHistory = new Dictionary<Agent, List<float>>();
        actionsEpisodeHistory = new Dictionary<Agent, List<float>>();
        valuesEpisodeHistory = new Dictionary<Agent, List<float>>();
        visualEpisodeHistory = new Dictionary<Agent, List<List<float[,,]>>>();

        accumulatedRewards = new Dictionary<Agent, float>();
        episodeSteps = new Dictionary<Agent, int>();


        var brainParameters = BrainToTrain.brainParameters;
        Debug.Assert(brainParameters.vectorActionSpaceType == SpaceType.continuous, "TrainerPPOCMA only support continuous actions space");
        List<DataBuffer.DataInfo> allBufferData = new List<DataBuffer.DataInfo>() {
            new DataBuffer.DataInfo("Action", typeof(float), new int[] {brainParameters.vectorActionSize[0]}),
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


        dataBuffer = new DataBuffer(allBufferData.ToArray());
        dataBufferTrainingVariance = new DataBuffer(varHistroyCount * parametersPPO.bufferSizeForTrain, allBufferData.ToArray());
        //pretrainObservationDataCollect = new List<List<float>>();
        //for (int i = 0; i < brainParameters.vectorObservationSize * brainParameters.numStackedVectorObservations; ++i)
        //{
        //    pretrainObservationDataCollect.Add(new List<float>());
        //}
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
        iModelPPO.ClipEpsilonValue = parametersPPO.clipValueLoss;

        if (BrainToTrain == null)
        {
            Debug.LogError("Please assign this trainer to a Brain with CoreBrainInternalTrainable!");
        }
        if (isTraining)
        {
            modelRef.SetLearningRate(parameters.learningRate, 0);
            modelRef.SetLearningRate(parameters.learningRate, 1);
            modelRef.SetLearningRate(parameters.learningRate, 2);
        }
    }

    public override void IncrementStep()
    {
        if (GetStep() == 0 && isTraining)
        {
            float[] obsMeans = new float[BrainToTrain.brainParameters.vectorObservationSize * BrainToTrain.brainParameters.numStackedVectorObservations];
            float[] obsStds = new float[BrainToTrain.brainParameters.vectorObservationSize * BrainToTrain.brainParameters.numStackedVectorObservations];

            for (int i = 0; i < BrainToTrain.brainParameters.vectorObservationSize * BrainToTrain.brainParameters.numStackedVectorObservations; ++i)
            {
                obsMeans[i] = 0;// pretrainObservationDataCollect[i].ToArray().Mean();
                obsStds[i] = 1;// (float)pretrainObservationDataCollect[i].ToArray().StandardDeviation();
            }
            
            for (int i = 0; i < pretrainingBatches; ++i)
            {
                float loss = iModelPPO.Pretrain(1, 0, obsStds, obsMeans, pretrainingBatchSize)[0];
                if(i % pretrainingBatches/10 == 0)
                    Debug.Log("pretrainloss:" + loss);
            }


        }

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
        foreach (var agent in agents)
        {
            statesEpisodeHistory[agent].Clear();
            rewardsEpisodeHistory[agent].Clear();
            actionsEpisodeHistory[agent].Clear();
            valuesEpisodeHistory[agent].Clear();
            accumulatedRewards[agent] = 0;
            episodeSteps[agent] = 0;
            agent.AgentReset();
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
            valuesEpisodeHistory[agent].Add(actionOutput[agent].value);

            //add the visual observations
            for (int i = 0; i < BrainToTrain.brainParameters.cameraResolutions.Length; ++i)
            {
                visualEpisodeHistory[agent][i].Add(currentInfo[agent].visualObservations[i]);
            }


            accumulatedRewards[agent] += newInfo[agent].reward;
            //print(accumulatedRewards[agent]);
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
                dataToAdd.Add(ValueTuple.Create<string, Array>("TargetValue", targetValues));
                dataToAdd.Add(ValueTuple.Create<string, Array>("OldValue", valueHistory));
                dataToAdd.Add(ValueTuple.Create<string, Array>("Advantage", advantages));
                if (statesEpisodeHistory[agent].Count > 0)
                    dataToAdd.Add(ValueTuple.Create<string, Array>("VectorObservation", statesEpisodeHistory[agent].ToArray()));
                for (int i = 0; i < visualEpisodeHistory[agent].Count; ++i)
                {
                    dataToAdd.Add(ValueTuple.Create<string, Array>("VisualObservation" + i, DataBuffer.ListToArray(visualEpisodeHistory[agent][i])));
                }

                dataBuffer.AddData(dataToAdd.ToArray());
                dataBufferTrainingVariance.AddData(dataToAdd.ToArray());
                //clear the temperary data record
                statesEpisodeHistory[agent].Clear();
                rewardsEpisodeHistory[agent].Clear();
                actionsEpisodeHistory[agent].Clear();
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

    public override Dictionary<Agent, TakeActionOutput> TakeAction(Dictionary<Agent, AgentInfoInternal> agentInfos)
    {
        var result = new Dictionary<Agent, TakeActionOutput>();
        var agentList = new List<Agent>(agentInfos.Keys);
        if (agentList.Count <= 0)
            return result;
        float[,] vectorObsAll = CreateVectorInputBatch(agentInfos, agentList);
        var visualObsAll = CreateVisualInputBatch(agentInfos, agentList, BrainToTrain.brainParameters.cameraResolutions);


        var values = iModelPPO.EvaluateValue(vectorObsAll, visualObsAll);
        var actions = iModelPPO.EvaluateAction(vectorObsAll, visualObsAll);

        /*if (GetStep() < pretrainDataCollectingSteps)
        {
            //collecting data for pretrain
            for (int n = 0; n < vectorObsAll.GetLength(1); ++n)
            {
                for (int m = 0; m < vectorObsAll.GetLength(0); ++m)
                {
                    pretrainObservationDataCollect[n].Add(vectorObsAll[m, n]);
                }
            }
        }*/

        int i = 0;
        foreach (var agent in agentList)
        {
            var agentDecision = agent.GetComponent<AgentDependentDecision>();

            if (isTraining && agentDecision != null && agentDecision.useDecision)// && UnityEngine.Random.Range(0, 1.0f) <= parametersPPO.useHeuristicChance)
            {
                //if this agent will use the decision, use it
                var info = agentInfos[agent];
                var action = agentDecision.Decide(info.stackedVectorObservation, info.visualObservations, new List<float>(actions.GetRow(i)));
                float[,] vectorOb = CreateVectorInputBatch(agentInfos, new List<Agent>() { agent });
                var visualOb = CreateVisualInputBatch(agentInfos, new List<Agent>() { agent }, BrainToTrain.brainParameters.cameraResolutions);

                var temp = new TakeActionOutput();
                temp.outputAction = action;
                temp.value = values[i];
                result[agent] = temp;
            }
            else
            {
                var temp = new TakeActionOutput();
                //temp.allProbabilities = actionProbs.GetRow(i);
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
        fetches.Add(new ValueTuple<string, int, string>("TargetValue", 0, "TargetValue"));
        fetches.Add(new ValueTuple<string, int, string>("OldValue", 0, "OldValue"));
        fetches.Add(new ValueTuple<string, int, string>("Advantage", 0, "Advantage"));
        for (int i = 0; i < BrainToTrain.brainParameters.cameraResolutions.Length; ++i)
        {
            fetches.Add(new ValueTuple<string, int, string>("VisualObservation" + i, 0, "VisualObservation" + i));
        }


        float policyMeanLoss = 0, valueLoss = 0, policyVarLoss = 0;
        for (int i = 0; i < parametersPPO.numEpochPerTrain; ++i)
        {
            //training from the main data buffer
            var samples = dataBuffer.SampleBatchesReordered(parametersPPO.batchSize, fetches.ToArray());

            var vectorObsArray = samples.TryGetOr("VectorObservation", null);
            float[,] vectorObservations = vectorObsArray == null ? null : vectorObsArray as float[,];
            float[,] actions = (float[,])samples["Action"];
            float[,] targetValues = (float[,])samples["TargetValue"];
            float[,] oldValues = (float[,])samples["OldValue"];
            float[] advantages = ((float[,])samples["Advantage"]).Flatten();


            //print("Adv before normalizatoin:" + advantages.Mean());

            float advMean = advantages.Mean();
            double advstd = advantages.StandardDeviation();
            for (int n = 0; n < advantages.Length; ++n)
            {
                // advantages[n] = (advantages[n] - advMean) / ((float)advstd + 0.0000000001f);
                advantages[n] = (advantages[n]) / ((float)advstd + 0.0000000001f);
            }



            List<float[,,,]> visualObservations = null;
            for (int j = 0; j < BrainToTrain.brainParameters.cameraResolutions.Length; ++j)
            {
                if (j == 0)
                    visualObservations = new List<float[,,,]>();
                visualObservations.Add((float[,,,])samples["VisualObservation" + j]);
            }

            int batchCount = targetValues.Length / parametersPPO.batchSize;
            //int actionUnitSize = (BrainToTrain.brainParameters.vectorActionSpaceType == SpaceType.continuous ? BrainToTrain.brainParameters.vectorActionSize : 1);
            //int totalStateSize = BrainToTrain.brainParameters.vectorObservationSize * BrainToTrain.brainParameters.numStackedVectorObservations;

            float tempPolicyMeanLoss = 0, tempPolicyVarLoss = 0, tempValueLoss = 0;
            for (int j = 0; j < batchCount; ++j)
            {
                var subVectorObs = SubRows(vectorObservations, j * parametersPPO.batchSize, parametersPPO.batchSize);
                var subVisualObs = SubRows(visualObservations, j * parametersPPO.batchSize, parametersPPO.batchSize);
                var subTargetValues = SubRows(targetValues, j * parametersPPO.batchSize, parametersPPO.batchSize).Flatten();
                var subOldValues = SubRows(oldValues, j * parametersPPO.batchSize, parametersPPO.batchSize).Flatten();
                tempValueLoss += iModelPPO.TrainValue(subVectorObs, subVisualObs, subOldValues, subTargetValues)[0];
            }

            for (int j = 0; j < batchCount; ++j)
            {
                var samplesVarTrain = dataBufferTrainingVariance.RandomSample(parametersPPO.batchSize, fetches.ToArray());

                var subVectorObsArray = samplesVarTrain.TryGetOr("VectorObservation", null);
                float[,] subVectorObservations = subVectorObsArray == null ? null : subVectorObsArray as float[,];
                float[,] subActions = (float[,])samplesVarTrain["Action"];
                float[] subAdvantages = ((float[,])samplesVarTrain["Advantage"]).Flatten();

                //float subAdvMean = subAdvantages.Mean();
                //double subAdvstd = subAdvantages.StandardDeviation();
                for (int n = 0; n < subAdvantages.Length; ++n)
                {
                    //subAdvantages[n] = (subAdvantages[n] - subAdvMean) / ((float)subAdvstd + 0.0000000001f);
                    subAdvantages[n] = (subAdvantages[n]) / ((float)advstd + 0.0000000001f);
                }



                List<float[,,,]> subVisualObservations = null;
                for (int v = 0; v < BrainToTrain.brainParameters.cameraResolutions.Length; ++v)
                {
                    if (v == 0)
                        subVisualObservations = new List<float[,,,]>();
                    subVisualObservations.Add((float[,,,])samplesVarTrain["VisualObservation" + v]);
                }


                tempPolicyVarLoss += iModelPPO.TrainVariance(subVectorObservations, subVisualObservations, subActions, subAdvantages)[0];
            }

            for (int j = 0; j < batchCount; ++j)
            {
                var subVectorObs = SubRows(vectorObservations, j * parametersPPO.batchSize, parametersPPO.batchSize);
                var subVisualObs = SubRows(visualObservations, j * parametersPPO.batchSize, parametersPPO.batchSize);
                var subOldActions = SubRows(actions, j * parametersPPO.batchSize, parametersPPO.batchSize);
                var subAdv = advantages.Get(j * parametersPPO.batchSize, (j + 1) * parametersPPO.batchSize);
                tempPolicyMeanLoss += iModelPPO.TrainMean(subVectorObs, subVisualObs, subOldActions, subAdv)[0];
            }


            policyMeanLoss += tempPolicyMeanLoss / (batchCount);
            valueLoss += tempValueLoss / (batchCount);
            policyVarLoss += tempPolicyVarLoss / (batchCount);
        }

        //log the stats
        stats.AddData("policyVarLoss", policyVarLoss / parametersPPO.numEpochPerTrain);
        stats.AddData("valueLoss", valueLoss / parametersPPO.numEpochPerTrain);
        stats.AddData("policyMeanLoss", policyMeanLoss / parametersPPO.numEpochPerTrain);


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

}
