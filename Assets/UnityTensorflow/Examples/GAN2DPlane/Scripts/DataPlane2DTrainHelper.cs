using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DataPlane2DTrainHelper : MonoBehaviour {


    public GANTrainHelper trainHelperRef;
    public GANModel modelRef;
    public DataPlane2D dataPlane;
    //public float lrGenerator = 0.001f;
    //public float lrDiscriminator = 0.001f;
    public bool training = false;
    public bool usePredictionInTraining = true;

    public int trainedEpisodes = 0;

    public float generatorLR = 0.001f;
    public float discriminatorLR = 0.001f;

    public int tryGanInterval = 50;
    protected int counter = 0;

    // Update is called once per frame
    void Update()
    {
        if (training)
        {
            modelRef.DiscriminatorLR = discriminatorLR;
            modelRef.GeneratorLR = generatorLR;
            TrainOnce(10);
            counter++;
            counter = counter % tryGanInterval;
            if(counter == 0)
            {
                UseGAN(800);
            }
        }


    }

    public void SetIfTraining(bool training)
    {
        this.training = training;
    }

    public void LoadTrainingData()
    {
        trainHelperRef.ClearData();
        trainHelperRef.AddData(null, dataPlane.GetDataPositions());
    }


    public void TrainOnce(int episodes)
    {
        float gloss = 0, dloss = 0;
        for (int i = 0; i < episodes; ++i)
        {
            var losses = trainHelperRef.TrainBothBatch(32,usePrediction:usePredictionInTraining);
            dloss += losses.Item1;
            gloss += losses.Item2;

            trainedEpisodes++;
        }

        Debug.Log("G loss: " + gloss/ episodes);
        Debug.Log("D loss: " + dloss/episodes);
    }

    public void UseGAN(int generatedNumber)
    {
        float[,] generated = (float[,])modelRef.GenerateBatch(null, MathUtils.GenerateWhiteNoise(generatedNumber, -1f, 1f, modelRef.inputNoiseShape));

        dataPlane.RemovePointsOfType(1);
        for (int i = 0; i < generatedNumber; ++i)
        {
            
            dataPlane.AddDatapoint(new Vector2(generated[i,0], generated[i,1]), 1);
        }

    }
    
}
