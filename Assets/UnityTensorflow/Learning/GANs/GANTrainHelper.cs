using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;



/// <summary>
/// A helper class to train GAN model if you want to use it without ML agent
/// </summary>
public class GANTrainHelper : MonoBehaviour
{

    public GANModel ganReference;


    public int maxDataBufferCount = 50000;
    protected DataBuffer dataBuffer = null;


    public void AddData(Array inputConditions, Array inputTargets)
    {

        if (dataBuffer == null)
        {
            //create databuffer if not exist yet
            List<DataBuffer.DataInfo> dataInfos = new List<DataBuffer.DataInfo>();

            if (ganReference.HasConditionInput)
            {
                dataInfos.Add(new DataBuffer.DataInfo("Condition", typeof(float), ganReference.inputConditionShape));
            }
            dataInfos.Add(new DataBuffer.DataInfo("Target", typeof(float), ganReference.outputShape));
            dataBuffer = new DataBuffer(maxDataBufferCount, dataInfos.ToArray());
        }

        //I am not checking the data size here because the dataBuffer.AddData will check it for me....tooo lazy
        List<ValueTuple<string, Array>> data = new List<ValueTuple<string, Array>>();
        if (ganReference.HasConditionInput)
        {
            data.Add(new ValueTuple<string, Array>("Condition", inputConditions));
        }
        data.Add(new ValueTuple<string, Array>("Target", inputTargets));
        dataBuffer.AddData(data.ToArray());
    }

    //clear all data
    public void ClearData()
    {
        if (dataBuffer != null)
            dataBuffer.ClearData();
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="batchSize"></param>
    /// <param name="generatorTrainCount"></param>
    /// <param name="discriminatorTrainCount"></param>
    /// <param name="usePrediction">https://openreview.net/pdf?id=Skj8Kag0Z</param>
    /// <returns>discriminator loss, generator loss</returns>
    public ValueTuple<float, float> TrainBothBatch(int batchSize, int generatorTrainCount = 1, int discriminatorTrainCount = 1, bool usePrediction = false)
    {
        float disLoss = 0, genLoss = 0;
        for (int i = 0; i < discriminatorTrainCount; ++i)
        {
            disLoss += TrainDiscriminatorBatch(batchSize);
        }

        if(usePrediction)
            ganReference.PredictWeights();

        for (int i = 0; i < generatorTrainCount; ++i)
        {
            genLoss += TrainGeneratorBatch(batchSize);
        }

        if (usePrediction)
            ganReference.RestoreFromPredictedWeights();

        return ValueTuple.Create(disLoss / discriminatorTrainCount, genLoss / generatorTrainCount);
    }




    public float TrainDiscriminatorBatch(int batchSize)
    {
        //fetch data from data buffer
        var fetchesForFake = new List<ValueTuple<string, int, string>>();
        var fetchesForReal = new List<ValueTuple<string, int, string>>();
        fetchesForReal.Add(new ValueTuple<string, int, string>("Target", 0, "Target"));
        if (ganReference.HasConditionInput)
        {
            fetchesForFake.Add(new ValueTuple<string, int, string>("Condition", 0, "Condition"));
            fetchesForReal.Add(new ValueTuple<string, int, string>("Condition", 0, "Condition"));
        }
        var samplesForFake = dataBuffer.RandomSample(batchSize, fetchesForFake.ToArray());
        var samplesForReal = dataBuffer.RandomSample(batchSize, fetchesForReal.ToArray());

        Array conditionsForFake = samplesForFake.ContainsKey("Condition") ? samplesForFake["Condition"] : null;
        Array conditionsForReal = samplesForReal.ContainsKey("Condition") ? samplesForReal["Condition"] : null;
        Array targetsForReal = samplesForReal["Target"];

        //generate noise
        Array noise = null;
        if (ganReference.HasNoiseInput)
        {
            noise = MathUtils.GenerateWhiteNoise(batchSize, -1, 1, ganReference.inputNoiseShape);
        }
        //genrate fake data
        Array generatedFake = ganReference.GenerateBatch(conditionsForFake, noise);

        //genreate labels
        float[,] fakeLabels = new float[batchSize, 1];
        float[,] realLabels = new float[batchSize, 1];
        for (int i = 0; i < batchSize; ++i)
        {
            fakeLabels[i, 0] = 0;
            realLabels[i, 0] = 1;
        }

        //train with fake labels
        float l1 = ganReference.TrainDiscriminatorBatch(generatedFake, fakeLabels, conditionsForFake);
        float l2 = ganReference.TrainDiscriminatorBatch(targetsForReal, realLabels, conditionsForReal);
        return (l1 + l2) / 2;
    }



    public float TrainGeneratorBatch(int batchSize)
    {

        //fetch data from data buffer
        var fetches = new List<ValueTuple<string, int, string>>();
        fetches.Add(new ValueTuple<string, int, string>("Target", 0, "Target"));
        if (ganReference.HasConditionInput)
        {
            fetches.Add(new ValueTuple<string, int, string>("Condition", 0, "Condition"));
        }
        var samples = dataBuffer.RandomSample(batchSize, fetches.ToArray());

        Array conditions = samples.ContainsKey("Condition") ? samples["Condition"] : null;
        Array targets = samples["Target"];

        Array noise = null;
        if (ganReference.HasNoiseInput)
        {
            noise = MathUtils.GenerateWhiteNoise(batchSize, -1, 1, ganReference.inputNoiseShape);
        }

        return ganReference.TrainGeneratorBatch(conditions, noise, targets);
    }

    public void SetGeneratorLR()
    {

    }

    public void SetDiscriminatorLR()
    {

    }
}
