using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Accord;
using Accord.Math;

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
