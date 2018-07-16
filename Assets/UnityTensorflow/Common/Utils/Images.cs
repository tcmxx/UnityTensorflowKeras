using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Threading;
using System;
using System.Threading.Tasks;


public static partial class Images
{
    public static byte[] GetRGB24FromTexture2D(this Texture2D texture)
    {
        Color[] pixels = texture.GetPixels();

        Texture2D temp = new Texture2D(texture.width, texture.height, TextureFormat.RGB24, false);
        temp.SetPixels(pixels);
        temp.Apply();
        return temp.GetRawTextureData();
    }

    public static byte[] GetRGB24FromTexture2D(this Texture2D texture, Vector2Int size)
    {
        Color[] pixels = texture.GetPixels();

        Texture2D temp = new Texture2D(texture.width, texture.height, TextureFormat.RGB24, false);
        temp.SetPixels(pixels);
        temp.Apply();
        TextureScale.Bilinear(temp, size.x, size.y);
        return temp.GetRawTextureData();
    }

    /// <summary>
    /// bytes of RGB24 without mipmap
    /// </summary>
    /// <param name="rgb24"></param>
    /// <returns></returns>
    public static float[] RGB24ToFloat(byte[] rgb24)
    {

        var result = new float[rgb24.Length];
        int numPix = rgb24.Length / 3;
        for (int i = 0; i < numPix; ++i)
        {
            result[i] = (float)rgb24[3 * i] / 255;
            result[i + numPix] = (float)rgb24[3 * i + 1] / 255;
            result[i + numPix * 2] = (float)rgb24[3 * i + 2] / 255;
        }
        return result;
    }

    /// <summary>
    /// bytes of RGB24 without mipmap
    /// </summary>
    /// <param name="rgb24"></param>
    /// <returns></returns>
    public static float[] RGB24ToFloat(IList<byte> rgb24)
    {

        var result = new float[rgb24.Count];
        int numPix = rgb24.Count / 3;
        for (int i = 0; i < numPix; ++i)
        {
            result[i] = (float)rgb24[3 * i] / 255;
            result[i + numPix] = (float)rgb24[3 * i + 1] / 255;
            result[i + numPix * 2] = (float)rgb24[3 * i + 2] / 255;
        }
        return result;
    }
    public static byte[] FloatToRGB24(float[] flattenedImageFloat)
    {
        var result = new byte[flattenedImageFloat.Length];
        int numPix = flattenedImageFloat.Length / 3;
        for (int i = 0; i < numPix; ++i)
        {
            result[3 * i] = (byte)(Mathf.Clamp01(flattenedImageFloat[i]) * 255);
            result[3 * i + 1] = (byte)(Mathf.Clamp01(flattenedImageFloat[numPix + i]) * 255);
            result[3 * i + 2] = (byte)(Mathf.Clamp01(flattenedImageFloat[2 * numPix + i]) * 255);
        }
        return result;
    }
    public static byte[] FloatToRGB24(IList<float> flattenedImageFloat)
    {
        var result = new byte[flattenedImageFloat.Count];
        int numPix = flattenedImageFloat.Count / 3;
        for (int i = 0; i < numPix; ++i)
        {
            result[3 * i] = (byte)(Mathf.Clamp01(flattenedImageFloat[i]) * 255);
            result[3 * i + 1] = (byte)(Mathf.Clamp01(flattenedImageFloat[numPix + i]) * 255);
            result[3 * i + 2] = (byte)(Mathf.Clamp01(flattenedImageFloat[2 * numPix + i]) * 255);
        }
        return result;
    }

    public static Color[] GenerateWhiteNoise(int length)
    {
        System.Random rd = new System.Random();
        var result = new Color[length];
        for (int i = 0; i < length; ++i)
        {
            result[i] = new Color(rd.Next(255) / 255.0f, rd.Next(255) / 255.0f, rd.Next(255) / 255.0f);
        }
        return result;
    }
    public static Color[] GeneratePureColor(int length, Color col)
    {
        var result = new Color[length];
        for (int i = 0; i < length; ++i)
        {
            result[i] = col;
        }
        return result;
    }

    public static Texture2D GetTextureWithAlpha(Texture2D colorSourceTexture, Texture2D alphaSourceTexture)
    {
        Texture2D tempAlpha = alphaSourceTexture;
        if (!(colorSourceTexture.height == alphaSourceTexture.height && colorSourceTexture.width == alphaSourceTexture.width))
        {
            //clone and scale the alpha source texture if needed
            tempAlpha = GetReadableTextureFromUnreadable(alphaSourceTexture);
            TextureScale.Bilinear(tempAlpha, colorSourceTexture.width, colorSourceTexture.height);
        }

        var result = new Texture2D(colorSourceTexture.width, colorSourceTexture.height);

        var originPixels = colorSourceTexture.GetPixels();
        var alphaPixels = tempAlpha.GetPixels();
        for (int i = 0; i < originPixels.Length; ++i)
        {
            var col = originPixels[i];
            col.a = alphaPixels[i].a;
            originPixels[i] = col;
        }
        result.SetPixels(originPixels);
        result.Apply();

        //destroy the temp texture if it is created
        if (tempAlpha != alphaSourceTexture)
        {
            GameObject.DestroyImmediate(tempAlpha);
        }
        return result;
    }


    public static Texture2D GetReadableTextureFromUnreadable(Texture2D texture)
    {
        // Create a temporary RenderTexture of the same size as the texture
        RenderTexture tmp = RenderTexture.GetTemporary(
                            texture.width,
                            texture.height,
                            0,
                            RenderTextureFormat.Default,
                            RenderTextureReadWrite.Linear);

        // Blit the pixels on texture to the RenderTexture
        Graphics.Blit(texture, tmp);
        // Backup the currently set RenderTexture
        RenderTexture previous = RenderTexture.active;
        // Set the current RenderTexture to the temporary one we created
        RenderTexture.active = tmp;
        // Create a new readable Texture2D to copy the pixels to it
        Texture2D myTexture2D = new Texture2D(texture.width, texture.height);
        // Copy the pixels from the RenderTexture to the new Texture
        myTexture2D.ReadPixels(new Rect(0, 0, tmp.width, tmp.height), 0, 0);
        myTexture2D.Apply();
        // Reset the active RenderTexture
        RenderTexture.active = previous;
        // Release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(tmp);

        return myTexture2D;
    }
}


/// <summary>
/// http://wiki.unity3d.com/index.php/TextureScale
/// </summary>
public class TextureScale
{
    public class ThreadData
    {
        public int start;
        public int end;
        public ThreadData(int s, int e)
        {
            start = s;
            end = e;
        }
    }

    private static Color[] texColors;
    private static Color[] newColors;
    private static int w;
    private static float ratioX;
    private static float ratioY;
    private static int w2;
    private static int finishCount;
    private static Mutex mutex;

    public static void Point(Texture2D tex, int newWidth, int newHeight)
    {
        ThreadedScale(tex, newWidth, newHeight, false);
    }

    public static void Bilinear(Texture2D tex, int newWidth, int newHeight)
    {
        ThreadedScale(tex, newWidth, newHeight, true);
    }

    private static void ThreadedScale(Texture2D tex, int newWidth, int newHeight, bool useBilinear)
    {
        texColors = tex.GetPixels();
        newColors = new Color[newWidth * newHeight];
        if (useBilinear)
        {
            ratioX = 1.0f / ((float)newWidth / (tex.width - 1));
            ratioY = 1.0f / ((float)newHeight / (tex.height - 1));
        }
        else
        {
            ratioX = ((float)tex.width) / newWidth;
            ratioY = ((float)tex.height) / newHeight;
        }
        w = tex.width;
        w2 = newWidth;
        var cores = Mathf.Min(SystemInfo.processorCount, newHeight);
        var slice = newHeight / cores;

        finishCount = 0;
        if (mutex == null)
        {
            mutex = new Mutex(false);
        }
        if (cores > 1)
        {
            int i = 0;
            ThreadData threadData;
            for (i = 0; i < cores - 1; i++)
            {
                threadData = new ThreadData(slice * i, slice * (i + 1));
                ParameterizedThreadStart ts = useBilinear ? new ParameterizedThreadStart(BilinearScale) : new ParameterizedThreadStart(PointScale);
                Thread thread = new Thread(ts);
                thread.Start(threadData);
            }
            threadData = new ThreadData(slice * i, newHeight);
            if (useBilinear)
            {
                BilinearScale(threadData);
            }
            else
            {
                PointScale(threadData);
            }
            while (finishCount < cores)
            {
                Thread.Sleep(1);
            }
        }
        else
        {
            ThreadData threadData = new ThreadData(0, newHeight);
            if (useBilinear)
            {
                BilinearScale(threadData);
            }
            else
            {
                PointScale(threadData);
            }
        }

        tex.Resize(newWidth, newHeight);
        tex.SetPixels(newColors);
        tex.Apply();

        texColors = null;
        newColors = null;
    }

    public static void BilinearScale(System.Object obj)
    {
        ThreadData threadData = (ThreadData)obj;
        for (var y = threadData.start; y < threadData.end; y++)
        {
            int yFloor = (int)Mathf.Floor(y * ratioY);
            var y1 = yFloor * w;
            var y2 = (yFloor + 1) * w;
            var yw = y * w2;

            for (var x = 0; x < w2; x++)
            {
                int xFloor = (int)Mathf.Floor(x * ratioX);
                var xLerp = x * ratioX - xFloor;
                newColors[yw + x] = ColorLerpUnclamped(ColorLerpUnclamped(texColors[y1 + xFloor], texColors[y1 + xFloor + 1], xLerp),
                                                       ColorLerpUnclamped(texColors[y2 + xFloor], texColors[y2 + xFloor + 1], xLerp),
                                                       y * ratioY - yFloor);
            }
        }

        mutex.WaitOne();
        finishCount++;
        mutex.ReleaseMutex();
    }

    public static void PointScale(System.Object obj)
    {
        ThreadData threadData = (ThreadData)obj;
        for (var y = threadData.start; y < threadData.end; y++)
        {
            var thisY = (int)(ratioY * y) * w;
            var yw = y * w2;
            for (var x = 0; x < w2; x++)
            {
                newColors[yw + x] = texColors[(int)(thisY + ratioX * x)];
            }
        }

        mutex.WaitOne();
        finishCount++;
        mutex.ReleaseMutex();
    }

    private static Color ColorLerpUnclamped(Color c1, Color c2, float value)
    {
        return new Color(c1.r + (c2.r - c1.r) * value,
                          c1.g + (c2.g - c1.g) * value,
                          c1.b + (c2.b - c1.b) * value,
                          c1.a + (c2.a - c1.a) * value);
    }
}

