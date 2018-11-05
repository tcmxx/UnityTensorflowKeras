using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Accord;
using System.Linq;

using MLAgents;

[Serializable]
public class EpisodeData : IComparable<EpisodeData>
{
    public float score;
    public bool isDone = false;
    public List<float> vectorObsHistory = null;
    public List<float> rewardsHistory = null;
    public List<float> actionsHistory = null;
    public List<List<float[,,]>> visualHistory = null;
    public List<List<float>> actionMasksHistory = null;
    public List<float> finalVectorObsHistory = null;
    public List<float[,,]> finalVisualObsHistory = null;

    public EpisodeData(List<float> vectorObs, List<float> rewards, List<float> actions, List<List<float[,,]>> visualObs, List<List<float>> actionMasks, List<float> finalVectorObs, List<float[,,]> finalVisualObs, bool isDone, float reward)
    {
        vectorObsHistory = new List<float>(vectorObs);
        rewardsHistory = new List<float>(rewards);
        actionsHistory = new List<float>(actions);

        visualHistory = visualObs.Select(x => new List<float[,,]>(x)).ToList();
        actionMasksHistory = actionMasks.Select(x => new List<float>(x)).ToList();
        finalVectorObsHistory = new List<float>(finalVectorObs);
        finalVisualObsHistory = new List<float[,,]>(finalVisualObs);
        this.isDone = isDone;

        this.score = reward;
    }

    public int CompareTo(EpisodeData obj)
    {
        return score.CompareTo(obj.score);
        /*if (ReferenceEquals(this,obj))
            return 0;
        if(reward < obj.reward)
        {
            return -1;
        }
        else
        {
            return 1;
        }*/
    }
}
[Serializable]
public class HPPORawHistory
{

    protected SortedSet<EpisodeData> episodesHistory;
    protected int maxSize = 0;
    protected float totalScore = 0;
    /*protected List<List<float>> vectorObsHistory = null;
    protected List<List<float>> rewardsHistory = null;
    protected List<List<float>> actionsHistory = null;
    protected List<List<List<float[,,]>>> visualHistory = null;
    protected List<List<List<float>>> actionMasksHistory = null;
    protected List<List<float>> finalVectorObsHistory = null;
    protected List<List<float[,,]>> finalVisualObsHistory = null;
    protected List<bool> isDoneHistory = null;*/
    public HPPORawHistory(int maxSize)
    {
        this.maxSize = maxSize;
        episodesHistory = new SortedSet<EpisodeData>();
    }
    public HPPORawHistory()
    {
        episodesHistory = new SortedSet<EpisodeData>();
        //episodesHistory.Add(n)
        /*vectorObsHistory = new List<List<float>>();
        rewardsHistory = new List<List<float>>();
        actionsHistory = new List<List<float>>();
        visualHistory = new List<List<List<float[,,]>>>();
        actionMasksHistory = new List<List<List<float>>>();
        finalVectorObsHistory = new List<List<float>>();
        finalVisualObsHistory = new List<List<float[,,]>>();
        isDoneHistory = new List<bool>();*/
    }


    public void AddEpisode(List<float> vectorObs, List<float> rewards, List<float> actions, List<List<float[,,]>> visualObs, List<List<float>> actionMasks, List<float> finalVectorObs, List<float[,,]> finalVisualObs, bool isDone)
    {
        /*vectorObsHistory.Add(new List<float>(vectorObs));
        rewardsHistory.Add(new List<float>(rewards));
        actionsHistory.Add(new List<float>(actions));
        
        visualHistory.Add(visualObs.Select(x => new List<float[,,]>(x)).ToList());
        actionMasksHistory.Add(actionMasks.Select(x => new List<float>(x)).ToList());
        finalVectorObsHistory.Add(new List<float>(finalVectorObs));
        finalVisualObsHistory.Add(new List<float[,,]>(finalVisualObs));
        isDoneHistory.Add(isDone);*/



        float score = rewards.Aggregate((x, y) => { return x + y; });
        if (maxSize > 0 && episodesHistory.Count >= maxSize)
        {
            if (score < totalScore / episodesHistory.Count)  //smaller than everage score
                return;

            var elementToRemove = episodesHistory.ElementAt(UnityEngine.Random.Range(0, episodesHistory.Count));
            totalScore += score;
            totalScore -= elementToRemove.score;
            episodesHistory.Remove(elementToRemove);
            episodesHistory.Add(new EpisodeData(vectorObs, rewards, actions, visualObs, actionMasks, finalVectorObs, finalVisualObs, isDone, score));

        }
        else
        {
            totalScore += score;
            episodesHistory.Add(new EpisodeData(vectorObs, rewards, actions, visualObs, actionMasks, finalVectorObs, finalVisualObs, isDone, score));
        }

    }

    public void Clear()
    {
        /*vectorObsHistory.Clear();
        rewardsHistory.Clear();
        actionsHistory.Clear();
        visualHistory.Clear();
        actionMasksHistory.Clear();
        finalVectorObsHistory.Clear();
        finalVisualObsHistory.Clear();
        isDoneHistory.Clear();*/

        episodesHistory.Clear();
    }


    public void EvaluateAndAddToDatabuffer(TrainerHPPO trainer, DataBuffer dataBuffer)
    {
        foreach (EpisodeData episode in episodesHistory)
        {
            float[] outValues, outTargetValues, outAdvantages;
            float[,] outAcionProbs;

            if (episode.rewardsHistory.Count > 0)
            {
                trainer.EvaluateEpisode(episode.vectorObsHistory, episode.visualHistory, episode.actionsHistory, episode.rewardsHistory, episode.actionMasksHistory,
                    out outValues, out outAcionProbs, out outTargetValues, out outAdvantages,
                    episode.isDone, episode.finalVectorObsHistory, episode.finalVisualObsHistory);

                List<ValueTuple<string, Array>> dataToAdd = new List<ValueTuple<string, Array>>();
                dataToAdd.Add(ValueTuple.Create<string, Array>("Action", episode.actionsHistory.ToArray()));
                dataToAdd.Add(ValueTuple.Create<string, Array>("ActionProb", outAcionProbs));
                dataToAdd.Add(ValueTuple.Create<string, Array>("TargetValue", outTargetValues));
                dataToAdd.Add(ValueTuple.Create<string, Array>("OldValue", outValues));
                dataToAdd.Add(ValueTuple.Create<string, Array>("Advantage", outAdvantages));
                if (episode.vectorObsHistory != null)
                    dataToAdd.Add(ValueTuple.Create<string, Array>("VectorObservation", episode.vectorObsHistory.ToArray()));
                for (int j = 0; j < episode.visualHistory.Count; ++j)
                {
                    dataToAdd.Add(ValueTuple.Create<string, Array>("VisualObservation" + j, DataBuffer.ListToArray(episode.visualHistory[j])));
                }
                for (int j = 0; j < episode.actionMasksHistory.Count; ++j)
                {
                    dataToAdd.Add(ValueTuple.Create<string, Array>("ActionMask" + j, episode.actionMasksHistory[j].ToArray()));
                }

                dataBuffer.AddData(dataToAdd.ToArray());
            }
        }
    }

    public DataBuffer AddToDataBuffer(BrainParameters brainParameter)
    {        

        List<DataBuffer.DataInfo> allBufferData = new List<DataBuffer.DataInfo>() {
            new DataBuffer.DataInfo("Action", typeof(float), new int[] { brainParameter.vectorActionSpaceType == SpaceType.continuous ? brainParameter.vectorActionSize[0] : brainParameter.vectorActionSize.Length })
        };

        if (brainParameter.vectorObservationSize > 0)
            allBufferData.Add(new DataBuffer.DataInfo("VectorObservation", typeof(float), new int[] { brainParameter.vectorObservationSize * brainParameter.numStackedVectorObservations }));

        for (int i = 0; i < brainParameter.cameraResolutions.Length; ++i)
        {
            int width = brainParameter.cameraResolutions[i].width;
            int height = brainParameter.cameraResolutions[i].height;
            int channels;
            if (brainParameter.cameraResolutions[i].blackAndWhite)
                channels = 1;
            else
                channels = 3;

            allBufferData.Add(new DataBuffer.DataInfo("VisualObservation" + i, typeof(float), new int[] { height, width, channels }));
        }
        allBufferData.Add(new DataBuffer.DataInfo("Reward", typeof(float), new int[] { 1 }));

        var dataBuffer = new DataBuffer(allBufferData.ToArray());

        foreach (EpisodeData episode in episodesHistory)
        {
            if (episode.rewardsHistory.Count > 0)
            {
                List<ValueTuple<string, Array>> dataToAdd = new List<ValueTuple<string, Array>>();
                dataToAdd.Add(ValueTuple.Create<string, Array>("Action", episode.actionsHistory.ToArray()));
                dataToAdd.Add(ValueTuple.Create<string, Array>("Reward", episode.rewardsHistory.ToArray()));

                if (episode.vectorObsHistory != null)
                    dataToAdd.Add(ValueTuple.Create<string, Array>("VectorObservation", episode.vectorObsHistory.ToArray()));
                for (int j = 0; j < episode.visualHistory.Count; ++j)
                {
                    dataToAdd.Add(ValueTuple.Create<string, Array>("VisualObservation" + j, DataBuffer.ListToArray(episode.visualHistory[j])));
                }
                for (int j = 0; j < episode.actionMasksHistory.Count; ++j)
                {
                    dataToAdd.Add(ValueTuple.Create<string, Array>("ActionMask" + j, episode.actionMasksHistory[j].ToArray()));
                }
                
                dataBuffer.AddData(dataToAdd.ToArray());
            }
        }
        return dataBuffer;
    }
}
