using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;

[Serializable]
public class DataBuffer : ISerializable
{
    [Serializable]
    public struct DataInfo
    {
        public DataInfo(string name, Type type, int[] dimension)
        {
            this.type = type;
            this.dimension = dimension;
            this.name = name;
            this.unitLength = dimension.Aggregate((t, a) => t * a);
        }
        public string name;
        public Type type;
        public int[] dimension;
        public int unitLength;
    }

    [Serializable]
    protected struct DataContainer
    {
        public DataContainer(DataInfo info, int maxDataCount)
        {
            this.info = info;
            dataList = Array.CreateInstance(info.type, (new int[] { maxDataCount }).Concat(info.dimension).ToArray());
        }
        public DataInfo info;
        public Array dataList;
    }


    protected Dictionary<string, DataContainer> dataset;

    public int MaxCount { get; private set; } = 0;
    private int nextBufferPointer = 0;
    public int CurrentCount { get; private set; } = 0;



    public DataBuffer(int maxSize, params DataInfo[] dataInfos)
    {

        MaxCount = maxSize;
        dataset = new Dictionary<string, DataContainer>();


        foreach (var i in dataInfos)
        {
            Debug.Assert(!dataset.ContainsKey(i.name));
            dataset[i.name] = new DataContainer(i, maxSize);
        }
    }


    public void GetObjectData(SerializationInfo info, StreamingContext context)
    {
        info.AddValue("MaxCount", MaxCount);
        info.AddValue("nextBufferPointer", nextBufferPointer);
        info.AddValue("CurrentCount", CurrentCount);
        info.AddValue("dataset", dataset);
    }
    // The special constructor is used to deserialize values.
    public DataBuffer(SerializationInfo info, StreamingContext context)
    {
        // Reset the property value using the GetValue method.
        MaxCount = (int)info.GetValue("MaxCount", typeof(int));
        nextBufferPointer = (int)info.GetValue("nextBufferPointer", typeof(int));
        CurrentCount = (int)info.GetValue("CurrentCount", typeof(int));
        dataset = (Dictionary<string, DataContainer>)info.GetValue("dataset", typeof(Dictionary<string, DataContainer>));
    }
    /// <summary>
    /// Add data to the buffer
    /// </summary>
    /// <param name="data"></param>
    public void AddData(params ValueTuple<string, Array>[] data)
    {
        //check whether the input data are correct
        Debug.Assert(data.Length == dataset.Count, "Input data number is not the same as the buffer required");
        int size = data[0].Item2.Length / dataset[data[0].Item1].info.unitLength;
        foreach (var k in dataset.Keys)
        {
            bool found = false;
            foreach (var d in data)
            {
                if (d.Item1.Equals(k))
                {
                    found = true;
                    int newSize = d.Item2.Length / dataset[d.Item1].info.unitLength;
                    Debug.Assert(newSize == size, "The input Data has different sizes");
                }
            }
            Debug.Assert(found == true, "Data " + k + " is not fed to the buffer");
        }

        //feed the data.

        //add the episode to the buffer
        int numToAdd = size;
        int spaceLeft = MaxCount - nextBufferPointer;

        int appendSize = Mathf.Min(spaceLeft, numToAdd);
        int fromStartSize = Mathf.Max(0, numToAdd - spaceLeft);

        foreach (var k in data)
        {
            DataContainer dd = dataset[k.Item1];
            int typeSize = Marshal.SizeOf(dd.info.type);
            //Debug.Log(k.Item1);
            //Debug.Log("add length " + k.Item2.Length + " copy length " + (appendSize * dd.info.unitLength).ToString());
            //Array.Copy(k.Item2, 0, dd.dataList, nextBufferPointer * dd.info.unitLength, appendSize * dd.info.unitLength);
            Buffer.BlockCopy(k.Item2, 0, dd.dataList, nextBufferPointer * dd.info.unitLength * typeSize, appendSize * dd.info.unitLength * typeSize);
        }
        nextBufferPointer += appendSize;
        CurrentCount += numToAdd;
        CurrentCount = Mathf.Clamp(CurrentCount, 0, MaxCount);
        if (fromStartSize > 0)
        {
            foreach (var k in data)
            {
                DataContainer dd = dataset[k.Item1];
                int typeSize = Marshal.SizeOf(dd.info.type);
                //Array.Copy(k.Item2, appendSize * dd.info.unitLength, dd.dataList, 0, fromStartSize * dd.info.unitLength);

                Buffer.BlockCopy(k.Item2, appendSize * dd.info.unitLength * typeSize, dd.dataList, 0, fromStartSize * dd.info.unitLength * typeSize);
            }
            nextBufferPointer = fromStartSize;
        }

    }

    public void ClearData()
    {
        nextBufferPointer = 0;
        CurrentCount = 0;
    }

    public Type GetDataType(string key)
    {
        return dataset[key].info.type;
    }

    /// <summary>
    /// get samples form the buffer. Might have repeated data.
    /// </summary>
    /// <param name="numOfSamples"></param>
    /// <param name="fetchAndOffset">tuple of <key of data to sample, sample index offset, returned dictionary key></param>
    /// <returns></returns>
    public Dictionary<string, Array> RandomSample(int numOfSamples, params ValueTuple<string, int, string>[] fetchAndOffset)
    {
        Debug.Assert(numOfSamples <= CurrentCount, "Not enough data to sample");

        Dictionary<string, Array> result = new Dictionary<string, Array>();

        foreach (var d in fetchAndOffset)
        {
            Debug.Assert(dataset.ContainsKey(d.Item1));
            Debug.Assert(!result.ContainsKey(d.Item3));
            //result[d.Item3] = Array.CreateInstance(GetDataType(d.Item1), dataset[d.Item1].info.unitLength * numOfSamples);
            result[d.Item3] = Array.CreateInstance(GetDataType(d.Item1), (new int[] { numOfSamples }).Concat(dataset[d.Item1].info.dimension).ToArray());
        }

        for (int i = 0; i < numOfSamples; ++i)
        {
            int sampleInd = UnityEngine.Random.Range(0, CurrentCount);
            foreach (var d in fetchAndOffset)
            {
                DataContainer c = dataset[d.Item1];
                int typeSize = Marshal.SizeOf(c.info.type);

                int unitLength = c.info.unitLength;
                int actSampleInd = (sampleInd + d.Item2) % CurrentCount;
                //Array.Copy(c.dataList, actSampleInd * unitLength, result[d.Item3], i * unitLength, unitLength);
                Buffer.BlockCopy(c.dataList, actSampleInd * unitLength * typeSize, result[d.Item3], i * unitLength * typeSize, unitLength * typeSize);
            }
        }

        return result;
    }

    /// <summary>
    /// get data at specific position
    /// </summary>
    /// <param name="index"></param>
    /// <param name="fetchAndOffset"></param>
    /// <returns></returns>
    public Dictionary<string, Array> FetchDataAt(int index, params ValueTuple<string, int, string>[] fetchAndOffset)
    {
        Debug.Assert(index < CurrentCount, "index out of bound");

        Dictionary<string, Array> result = new Dictionary<string, Array>();

        foreach (var d in fetchAndOffset)
        {
            Debug.Assert(dataset.ContainsKey(d.Item1));
            Debug.Assert(!result.ContainsKey(d.Item3));
            //result[d.Item3] = Array.CreateInstance(GetDataType(d.Item1), dataset[d.Item1].info.unitLength * numOfSamples);
            result[d.Item3] = Array.CreateInstance(GetDataType(d.Item1), (new int[] { 1 }).Concat(dataset[d.Item1].info.dimension).ToArray());
        }

        foreach (var d in fetchAndOffset)
        {
            DataContainer c = dataset[d.Item1];
            int typeSize = Marshal.SizeOf(c.info.type);

            int unitLength = c.info.unitLength;
            int actSampleInd = (index + d.Item2) % CurrentCount;
            //Array.Copy(c.dataList, actSampleInd * unitLength, result[d.Item3], i * unitLength, unitLength);
            Buffer.BlockCopy(c.dataList, actSampleInd * unitLength * typeSize, result[d.Item3], 0, unitLength * typeSize);
        }


        return result;
    }



    /// <summary>
    /// sample as many batches as possible with data reordered. No repeated data.
    /// </summary>
    /// <param name="batchSize"></param>
    /// <param name="fetchAndOffset"></param>
    /// <returns></returns>
    public Dictionary<string, Array> SampleBatchesReordered(int batchSize,  params ValueTuple<string, int, string>[] fetchAndOffset)
    {
        return SampleBatchesReordered(batchSize, -1, fetchAndOffset);
    }
    /// <summary>
    /// sample as many batches as maxBatchesCount with data reordered. No repeated data.
    /// </summary>
    /// <param name="batchSize"></param>
    /// /// <param name="maxBatchesCount">max number of batches to sample. if it is less than one, it will try to get as many as possible</param>
    /// <param name="fetchAndOffset"></param>
    /// <returns></returns>
    public Dictionary<string, Array> SampleBatchesReordered(int batchSize, int maxBatchesCount = -1, params ValueTuple<string, int, string>[] fetchAndOffset)
    {
        //Debug.Assert(batchSize <= CurrentCount, "Not enough data to sample");

        Dictionary<string, Array> result = new Dictionary<string, Array>();



        int numToSample = ((int)(CurrentCount / batchSize)) * batchSize;
        int[] indices = new int[numToSample];
        for (int i = 0; i < numToSample; ++i)
        {
            indices[i] = i;
        }
        MathUtils.Shuffle(indices, new System.Random());

        foreach (var d in fetchAndOffset)
        {
            Debug.Assert(dataset.ContainsKey(d.Item1));
            Debug.Assert(!result.ContainsKey(d.Item3));
            //result[d.Item3] = Array.CreateInstance(GetDataType(d.Item1), dataset[d.Item1].info.unitLength * numToSample);
            result[d.Item3] = Array.CreateInstance(GetDataType(d.Item1), (new int[] { numToSample }).Concat(dataset[d.Item1].info.dimension).ToArray());
        }

        if (maxBatchesCount > 0)
            numToSample = Mathf.Min(numToSample, maxBatchesCount);
        for (int i = 0; i < numToSample; ++i)
        {
            int sampleInd = indices[i];
            foreach (var d in fetchAndOffset)
            {
                DataContainer c = dataset[d.Item1];
                int typeSize = Marshal.SizeOf(c.info.type);
                int unitLength = c.info.unitLength;
                int actSampleInd = (sampleInd + d.Item2) % CurrentCount;
                //Array.Copy(c.dataList, actSampleInd * unitLength, result[d.Item3], i * unitLength, unitLength);
                Buffer.BlockCopy(c.dataList, actSampleInd * unitLength * typeSize, result[d.Item3], i * unitLength * typeSize, unitLength * typeSize);
            }
        }
        return result;
    }

    /// <summary>
    /// calculate the discrounted reward
    /// </summary>
    /// <param name="stepReward">the rewards list of each step</param>
    /// <param name="gamma">discount factor</param>
    /// <param name="nextValue">The reward after the last step of the list.</param>
    /// <returns></returns>
    public static float[] GetDiscountedRewards(List<float> stepReward, float gamma, float nextValue = 0)
    {
        float accum = nextValue;
        float[] result = new float[stepReward.Count];
        for (int i = stepReward.Count - 1; i >= 0; --i)
        {
            accum = accum * gamma + stepReward[i];
            result[i] = accum;
        }

        return result;
    }


    /// <summary>
    /// Get the Advantage for PPO algorithm
    /// </summary>
    /// <param name="stepRewards"></param>
    /// <param name="valueEstimates"></param>
    /// <param name="gamma"></param>
    /// <param name="lambda"></param>
    /// <param name="nextValue"></param>
    /// <returns></returns>
    public static float[] GetGAE(List<float> stepRewards, List<float> valueEstimates, float gamma, float lambda, float nextValue = 0)
    {
        Debug.Assert(stepRewards.Count == valueEstimates.Count, "stepReward and valueEstimates need to have the same length");

        int length = stepRewards.Count;
        List<float> deltaTs = new List<float>(length);
        for (int i = 0; i < length - 1; ++i)
        {
            deltaTs.Add(stepRewards[i] + gamma * valueEstimates[i + 1] - valueEstimates[i]);
        }
        deltaTs.Add(stepRewards[length - 1] + gamma * nextValue - valueEstimates[length - 1]);

        float[] advantages = GetDiscountedRewards(deltaTs, gamma * lambda);
        return advantages;
    }



    public static float[,,,] ListToArray(List<float[,,]> arrays)
    {
        List<int> lengths = new List<int>();
        int totalLength = arrays[0].Length;
        lengths.Add(arrays.Count);
        for (int i = 0; i < arrays[0].Rank; ++i)
        {
            lengths.Add(arrays[0].GetLength(i));
        }
        var result = Array.CreateInstance(typeof(float), lengths.ToArray());

        int typeSize = sizeof(float);
        for (int i = 0; i < arrays.Count; ++i)
        {
            Debug.Assert(arrays[i].Length == totalLength, "Input arrays must have the same length");
            Buffer.BlockCopy(arrays[i], 0, result, i * totalLength * typeSize, totalLength * typeSize);
        }
        return result as float[,,,];
    }


}
