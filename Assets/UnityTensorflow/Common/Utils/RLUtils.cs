using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class RLUtils
{
    public static float[] DiscountedRewards(float[] rewards, float discountFactor = 0.99f, float nextValue = 0)
    {
        float accum = nextValue;
        float[] result = new float[rewards.Length];
        for (int i = rewards.Length - 1; i >= 0; --i)
        {
            accum = accum * discountFactor + rewards[i];
            result[i] = accum;
        }

        return result;
    }

    public static float[] GeneralAdvantageEst(float[] rewards, float[] estimatedValues, float discountedFactor = 0.99f, float GAEFactor = 0.95f, float nextValue = 0)
    {
        Debug.Assert(rewards.Length == estimatedValues.Length);
        float[] deltaT = new float[rewards.Length];
        for (int i = 0; i < rewards.Length; ++i)
        {
            if (i != rewards.Length - 1)
            {
                deltaT[i] = rewards[i] + discountedFactor * estimatedValues[i + 1] - estimatedValues[i];
            }
            else
            {
                deltaT[i] = rewards[i] + discountedFactor * nextValue - estimatedValues[i];
            }

        }
        return DiscountedRewards(deltaT, GAEFactor * discountedFactor);
    }

}