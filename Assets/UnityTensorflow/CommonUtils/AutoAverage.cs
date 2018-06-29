using System.Collections;
using System.Collections.Generic;
using UnityEngine;





/// <summary>
/// helps to get the average of data
/// </summary>
public class AutoAverage
{
    private int interval;
    public int Interval
    {
        get { return interval; }
        set { interval = Mathf.Max(value, 1); }
    }

    public float Average
    {
        get
        {
            return lastAverage;
        }
    }

    public bool JustUpdated
    {
        get; private set;
    }

    private float lastAverage = 0;
    private int currentCount = 0;
    private float sum = 0;

    public AutoAverage(int interval = 1)
    {
        Interval = interval;
        JustUpdated = false;
    }

    public void AddValue(float value)
    {
        sum += value;
        currentCount += 1;
        JustUpdated = false;
        if (currentCount >= Interval)
        {
            lastAverage = sum / currentCount;
            currentCount = 0;
            sum = 0;
            JustUpdated = true;
        }
    }


}

