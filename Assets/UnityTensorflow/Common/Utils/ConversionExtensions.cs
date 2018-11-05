using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Accord;
using Accord.Math;
using System.Runtime.InteropServices;

public static class ConversionExtensions {



    public static T[] FlattenAndConvertArray<T>(this Array data)
    {
        Array flattened = data.DeepFlatten();
        Type elementType = flattened.GetType().GetElementType();
        
        if(elementType == typeof(float))
        {
            return Array.ConvertAll((float[])flattened, t => (T)Convert.ChangeType(t,typeof(T)));
        }
        if (elementType == typeof(int))
        {
            return Array.ConvertAll((float[])flattened, t => (T)Convert.ChangeType(t, typeof(T)));
        }
        if (elementType == typeof(double))
        {
            return Array.ConvertAll((float[])flattened, t => (T)Convert.ChangeType(t, typeof(T)));
        }
        else
        {
            Debug.LogError("input array element type not supported");
            throw new NotImplementedException();
        }

        
    }

    public static T[,] Reshape<T>(this List<T> data, int colSize)
    {
        if (colSize == 0)
            return null;
        var result = new T[data.Count / colSize, colSize];
        int typeSize = Marshal.SizeOf(typeof(T));
        Buffer.BlockCopy(data.ToArray(), 0, result, 0, data.Count * typeSize);

        return result;
    }

    public static T[,,,] Stack<T>(this List<T[,,]> data)
    {
        int width = data[0].GetLength(0);
        int height = data[0].GetLength(1);
        int depth = data[0].GetLength(2);
        int eachDataLength = width* height* depth;
        var result = new T[data.Count, width, height, depth];
        int typeSize = Marshal.SizeOf(typeof(T));

        for (int i = 0; i < data.Count; ++i)
        {
            Buffer.BlockCopy(data[i], 0, result, i * eachDataLength * typeSize, eachDataLength * typeSize);
        }
        return result;

    }

    public static T[,] SubRows<T>(this T[,] data, int startRow, int rowCount)
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

    public static List<T[,,,]> SubRows<T>(this List<T[,,,]> data, int startRow, int rowCount)
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

    public static List<T[,]> SubRows<T>(this List<T[,]> data, int startRow, int rowCount)
    {
        if (data == null || data.Count == 0)
            return null;
        List<T[,]> result = new List<T[,]>();
        for (int i = 0; i < data.Count; ++i)
        {
            int rowLength1 = data[i].GetLength(1);
            int rowLengthTotal = rowLength1;

            result.Add(new T[rowCount, rowLength1]);
            int typeSize = Marshal.SizeOf(typeof(T));

            Buffer.BlockCopy(data[i], startRow * rowLengthTotal * typeSize, result[i], 0, rowCount * rowLengthTotal * typeSize);
        }

        return result;
    }

    /// <summary>
    /// return the 3D float array of the texture image.
    /// </summary>
    /// <param name="tex">texture</param>
    /// <param name="blackAndWhite">whether return black and white</param>
    /// <returns>HWC array of the image</returns>
    public static float[,,] TextureToArray(this Texture2D tex, bool blackAndWhite)
    {
        int width = tex.width;
        int height = tex.height;
        int pixels = 0;
        if (blackAndWhite)
            pixels = 1;
        else
            pixels = 3;
        float[,,] result = new float[height, width, pixels];
        float[] resultTemp = new float[height * width * pixels];
        int wp = width * pixels;

        Color32[] cc = tex.GetPixels32();
        for (int h = height - 1; h >= 0; h--)
        {
            for (int w = 0; w < width; w++)
            {
                Color32 currentPixel = cc[(height - h - 1) * width + w];
                if (!blackAndWhite)
                {
                    resultTemp[h * wp + w * pixels] = currentPixel.r / 255.0f;
                    resultTemp[h * wp + w * pixels + 1] = currentPixel.g / 255.0f;
                    resultTemp[h * wp + w * pixels + 2] = currentPixel.b / 255.0f;
                }
                else
                {
                    resultTemp[h * wp + w * pixels] =
                    (currentPixel.r + currentPixel.g + currentPixel.b)
                    / 3;
                }
            }
        }

        Buffer.BlockCopy(resultTemp, 0, result, 0, height * width * pixels * sizeof(float));
        return result;
    }

    /// <summary>
    /// Converts a list of Texture2D into a Tensor. Modified from the CoreBrainInternal.cs script
    /// </summary>
    /// <returns>
    /// A 4 dimensional float Tensor of dimension
    /// [batch_size, height, width, channel].
    /// Where batch_size is the number of input textures,
    /// height corresponds to the height of the texture,
    /// width corresponds to the width of the texture,
    /// channel corresponds to the number of channels extracted from the
    /// input textures (based on the input blackAndWhite flag
    /// (3 if the flag is false, 1 otherwise).
    /// The values of the Tensor are between 0 and 1.
    /// </returns>
    /// <param name="textures">
    /// The list of textures to be put into the tensor.
    /// Note that the textures must have same width and height.
    /// </param>
    /// <param name="blackAndWhite">
    /// If set to <c>true</c> the textures
    /// will be converted to grayscale before being stored in the tensor.
    /// </param>
    public static float[,,,] BatchVisualObservations(
        this List<Texture2D> textures, bool blackAndWhite)
    {
        int batchSize = textures.Count;
        int width = textures[0].width;
        int height = textures[0].height;
        int pixels = 0;
        if (blackAndWhite)
            pixels = 1;
        else
            pixels = 3;
        float[,,,] result = new float[batchSize, height, width, pixels];
        float[] resultTemp = new float[batchSize * height * width * pixels];
        int hwp = height * width * pixels;
        int wp = width * pixels;
        for (int b = 0; b < batchSize; b++)
        {
            Color32[] cc = textures[b].GetPixels32();
            for (int h = height - 1; h >= 0; h--)
            {
                for (int w = 0; w < width; w++)
                {
                    Color32 currentPixel = cc[(height - h - 1) * width + w];
                    if (!blackAndWhite)
                    {
                        // For Color32, the r, g and b values are between
                        // 0 and 255.
                        /*result[b, height - h - 1, w, 0] =
                            currentPixel.r / 255.0f;
                        result[b, height - h - 1, w, 1] =
                            currentPixel.g / 255.0f;
                        result[b, height - h - 1, w, 2] =
                            currentPixel.b / 255.0f;*/

                        resultTemp[b * hwp + h * wp + w * pixels] = currentPixel.r / 255.0f;
                        resultTemp[b * hwp + h * wp + w * pixels + 1] = currentPixel.g / 255.0f;
                        resultTemp[b * hwp + h * wp + w * pixels + 2] = currentPixel.b / 255.0f;

                        /*result[b,h,w,0] = currentPixel.r / 255.0f;
                        result[b, h, w, 1] = currentPixel.g / 255.0f;
                        result[b, h, w, 2] = currentPixel.b / 255.0f;*/
                    }
                    else
                    {
                        /*result[b, height - h - 1, w, 0] =
                            (currentPixel.r + currentPixel.g + currentPixel.b)
                            / 3;*/
                         resultTemp[b * hwp + h * wp + w * pixels] =
                             (currentPixel.r + currentPixel.g + currentPixel.b)
                             / 3;
                        /*result[b, h, w, 0] =
                             (currentPixel.r + currentPixel.g + currentPixel.b)
                             / 3;*/
                    }
                }
            }
        }

        Buffer.BlockCopy(resultTemp, 0, result, 0, batchSize * height * width * pixels * sizeof(float));

        return result;
    }
}
