using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BilliardGANVisualHelper : MonoBehaviour {
    public GANModel modelRef;
    public DataPlane2D dataPlane;
    public BilliardGameSystem gameSystem;


    public void UseGAN(int generatedNumber)
    {

        var balls = gameSystem.GetBallsStatus();

        float[,] conditionsAll = new float[generatedNumber, balls.Count * 3];

        for (int iball = 0; iball < balls.Count;++iball)
        {
            for(int i = 0; i < generatedNumber; ++i)
            {
                conditionsAll[i, iball*3] = balls[iball].x;
                conditionsAll[i, iball * 3+1] = balls[iball].y;
                conditionsAll[i, iball * 3+2] = balls[iball].z;
            }
        }

        float[,] generated = (float[,])modelRef.GenerateBatch(conditionsAll, MathUtils.GenerateWhiteNoise(generatedNumber, -1, 1, modelRef.inputNoiseShape));

        dataPlane.RemovePointsOfType(1);
        dataPlane.RemovePointsOfType(0);
        for (int i = 0; i < generatedNumber; ++i)
        {

            dataPlane.AddDatapoint(new Vector2(generated[i, 0]/2, generated[i, 1])/2, 1);
        }

        dataPlane.AddDatapoint(new Vector2(0.5f, 0.5f), 0);
        dataPlane.AddDatapoint(new Vector2(-0.5f, -0.5f), 0);
        dataPlane.AddDatapoint(new Vector2(0.5f, -0.5f), 0);
        dataPlane.AddDatapoint(new Vector2(-0.5f, 0.5f), 0);
        dataPlane.AddDatapoint(new Vector2(0, 0),0);

    }
}
