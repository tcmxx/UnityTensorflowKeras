using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace NWH
{
    public static class FileHandler
    {

        public static string defaultWritePath = Path.Combine(Directory.GetCurrentDirectory(),"PlotsLog");
       
        public static string delimiter = ",";

        public static void AppendSamplesToCSV(List<Sample> samples, string filename, string writePath = "")
        {
            string output = "";
            foreach (Sample s in samples)
            {
                output += s.t + delimiter + s.d + Environment.NewLine;
            }
            if (string.IsNullOrEmpty(writePath))
                writePath = defaultWritePath;
            Directory.CreateDirectory(writePath);
            File.AppendAllText(Path.Combine(writePath, filename), output);
        }

        public static List<Sample> LoadSamplesFromCSV(string filename, string writePath = "")
        {
            List<Sample> sampleList = new List<Sample>();

            if (string.IsNullOrEmpty(writePath))
                writePath = defaultWritePath;
            Directory.CreateDirectory(writePath);

            string fileData = File.ReadAllText(Path.Combine(writePath, filename));
            string[] lines = fileData.Split(new string[] { Environment.NewLine }, StringSplitOptions.None);

            for (int i = 1; i < lines.Length - 1; i++)
            {
                string[] vs = lines[i].Trim().Split(',');
                try
                {
                    Sample gs = new Sample(float.Parse(vs[1]), float.Parse(vs[0]));
                    sampleList.Add(gs);
                }
                catch { }
            }
            return sampleList;
        }

        public static string LoadHeaderFromCSV(string filename, string writePath = "")
        {
            if (string.IsNullOrEmpty(writePath))
                writePath = defaultWritePath;
            Directory.CreateDirectory(writePath);

            string fileData = File.ReadAllText(Path.Combine(writePath, filename));
            string[] lines = fileData.Split(new string[] { Environment.NewLine }, StringSplitOptions.None);

            if (lines.Length > 0)
            {
                return lines[0];
            }
            else
            {
                Debug.LogWarning("File you are trying to open is empty.");
                return null;
            }
        }

        public static void AppendStringToCSV(string input, string filename, string writePath = "")
        {
            if (string.IsNullOrEmpty(writePath))
                writePath = defaultWritePath;
            Directory.CreateDirectory(writePath);

            File.AppendAllText(Path.Combine(writePath, filename), input);
        }

        public static void WriteStringToCSV(string input, string filename, string writePath = "")
        {
            if (string.IsNullOrEmpty(writePath))
                writePath = defaultWritePath;
            Directory.CreateDirectory(writePath);

            File.WriteAllText(Path.Combine(writePath, filename), input);
        }


        public static string BrowserSaveFiles(string defaultName)
        {

            return EditorUtility.SaveFilePanel("Save to", defaultWritePath, defaultName,"ses");
        }
        public static List<string> BrowserOpenFiles(string writePath = "")
        {
            if (string.IsNullOrEmpty(writePath))
                writePath = defaultWritePath;
            Directory.CreateDirectory(writePath);

            string path = EditorUtility.OpenFilePanel("Open previous recording(s)", writePath, "");
            if (path == null || path == "")
            {
                return null;
            }
            else
            {
                if (Path.GetExtension(path) == ".csv")
                {
                    // Fetch single file
                    string[] arr = new string[1];
                    arr[0] = path;
                    Debug.Log("Loaded " + path);
                    return arr.ToList();
                }
                else if (Path.GetExtension(path) == ".ses")
                {
                    // Fetch all files for the recording
                    string fileData = File.ReadAllText(path);
                    List<string> filenameList = fileData.Split(new string[] { Environment.NewLine }, StringSplitOptions.None).ToList();

                    foreach (string s in filenameList)
                    {
                        Debug.Log("Loaded " + s);
                    }

                    return filenameList;
                }
                else
                {
                    // Unknown extension
                    Debug.LogError("Unknown file format. Only .csv and .ses files are supported");
                    return null;
                }
            }
        }

        public static string CleanFilename(string fileName)
        {
            return Path.GetInvalidFileNameChars().Aggregate(fileName, (current, c) => current.Replace(c.ToString(), string.Empty)).Replace(" ", "");
        }
    }
}