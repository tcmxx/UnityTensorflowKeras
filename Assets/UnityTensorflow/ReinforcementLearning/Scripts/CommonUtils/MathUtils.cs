using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public static class MathUtils
{


    public static float[] GenerateWhiteNoise(int size, float min, float max)
    {
        if (size <= 0)
            return null;
        float[] result = new float[size];
        for (int i = 0; i < size; ++i)
        {
            result[i] = UnityEngine.Random.Range(min, max);
        }
        return result;
    }


    public static float NextGaussianFloat()
    {
        float u, v, S;

        do
        {
            u = 2.0f * Random.value - 1.0f;
            v = 2.0f * Random.value - 1.0f;
            S = u * u + v * v;
        }
        while (S >= 1.0);

        float fac = Mathf.Sqrt(-2.0f * Mathf.Log(S) / S);
        return u * fac;
    }




    public enum InterpolateMethod
    {
        Linear,
        Log
    }

    /// <summary>
    /// interpolate between x1 and x2 to ty suing the interpolate method
    /// </summary>
    /// <param name="method"></param>
    /// <param name="x1"></param>
    /// <param name="x2"></param>
    /// <param name="t"></param>
    /// <returns></returns>
    public static float Interpolate(float x1, float x2, float t, InterpolateMethod method = InterpolateMethod.Linear)
    {
        if (method == InterpolateMethod.Linear)
        {
            return Mathf.Lerp(x1, x2, t);
        }
        else
        {
            return Mathf.Pow(x1, 1 - t) * Mathf.Pow(x2, t);
        }
    }

    /// <summary>
    /// Return a index randomly. The probability if a index depends on the value in that list
    /// </summary>
    /// <param name="list"></param>
    /// <returns></returns>
    public static int IndexByChance(IList<float> list)
    {
        float total = 0;

        foreach (var v in list)
        {
            total += v;
        }
        Debug.Assert(total > 0);

        float current = 0;
        float point = Random.Range(0, total);

        for (int i = 0; i < list.Count; ++i)
        {
            current += list[i];
            if (current >= point)
            {
                return i;
            }
        }
        return 0;
    }
    /// <summary>
    /// return the index of the max value in the list
    /// </summary>
    /// <param name="list"></param>
    /// <returns></returns>
    public static int IndexMax(IList<float> list)
    {
        int result = 0;
        float max = Mathf.NegativeInfinity;
        for (int i = 0; i < list.Count; ++i)
        {
            if (max < list[i])
            {
                result = i;
            }
        }
        return result;
    }

    /// <summary>
    /// Shuffle a list
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="list"></param>
    /// <param name="rnd"></param>
    public static void Shuffle<T>(IList<T> list, System.Random rnd)
    {
        int n = list.Count;
        while (n > 1)
        {

            n--;
            int k = rnd.Next(0, n + 1);
            T value = list[k];
            list[k] = list[n];
            list[n] = value;
        }
    }

}
