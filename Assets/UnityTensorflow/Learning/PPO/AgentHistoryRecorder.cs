using Accord.Math;
using MLAgents;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class AgentHistoryRecorder : MonoBehaviour {

    public int maxHistoryEpisode = 100;
    protected List< List<float>> vectorObsEpisodeHistory = null;
    protected List<List<float>> rewardsEpisodeHistory = null;
    protected List<List<float>> actionsEpisodeHistory = null;
    protected List<bool> failEpisodeHistory = null;
    protected List<List<List<float[,,]>>> visualObsEpisodeHistory = null;

    protected int currentEpisode = 0;

    protected Agent agentRef;

    protected int vectorObsSize;

    private void Awake()
    {
        vectorObsEpisodeHistory = new List<List<float>>();
        rewardsEpisodeHistory = new List<List<float>>();
        actionsEpisodeHistory = new List<List<float>>();
        failEpisodeHistory = new List<bool>();
        visualObsEpisodeHistory = new List<List<List<float[,,]>>>();

        agentRef = GetComponent<Agent>();
        Debug.Assert(agentRef != null, "AgentHistory need to be attached to a gameobject with Agent script"); 
    }
    // Use this for initialization
    void Start () {
        vectorObsSize = agentRef.brain.brainParameters.numStackedVectorObservations * agentRef.brain.brainParameters.vectorObservationSize;

    }
	

    public enum StepStatus
    {
        DoneWithFailure,
        DoneWithoutFailure,
        NotDone
    }

	/*public void PushRecord(float[] actions, float[] vectorObs, float reward, List<float[,,]> visualObs, StepStatus stepstatus, float[] finalVectorObsIfDoneWithoutFailure = null, List<float[,,]> finalVisualObsIfDoneWithoutFailure = null)
    {

        if(newEpisode && currentEpisode >= vectorObsEpisodeHistory.Count)
        {
            vectorObsEpisodeHistory.Add(new List<float>());
            rewardsEpisodeHistory.Add(new List<float>());
            actionsEpisodeHistory.Add(new List<float>());
            visualObsEpisodeHistory.Add(new List<List<float[,,]>>());
        }else if (newEpisode)
        {
            vectorObsEpisodeHistory[currentEpisode].Clear();
            rewardsEpisodeHistory[currentEpisode].Clear();
            actionsEpisodeHistory[currentEpisode].Clear();
            visualObsEpisodeHistory[currentEpisode].Clear();
        }

        vectorObsEpisodeHistory[currentEpisode].AddRange(vectorObs);
        rewardsEpisodeHistory[currentEpisode].Add(reward);
        actionsEpisodeHistory[currentEpisode].AddRange(actions);
        visualObsEpisodeHistory[currentEpisode].Add(visualObs);


        if (stepstatus != StepStatus.NotDone)
        {
            currentEpisode = maxHistoryEpisode > 0 ? (currentEpisode + 1) % maxHistoryEpisode : (currentEpisode + 1);
        }
    }*/


    /*DataBuffer ConvertHistoryForPPO(IRLModelPPO ppoModel)
    {
        for(int i = 0; i < vectorObsEpisodeHistory.Count;++i)
        {
            //update process the episode data for PPO.
            float nextValue = 0;
            int numOfSteps = rewardsEpisodeHistory[i].Count;
            if (failEpisodeHistory[i] == false)
            {   
                //this episode is not a failure
                nextValue = 0;  //this is very important!
            }
            else
            {
                nextValue = ppoModel.EvaluateValue(Matrix.Reshape(vectorObsEpisodeHistory[i].GetRange(vectorObsEpisodeHistory[i].Count - vectorObsSize, vectorObsSize).ToArray(), 1, vectorObsSize),
                    visualObsEpisodeHistory[i][numOfSteps-1].Select((t)=>  DataBuffer.ListToArray(new List<float[,,]> { t })).ToList())[0];
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
    }*/
}
