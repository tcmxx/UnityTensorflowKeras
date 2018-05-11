using System.Collections;
using System.Collections.Generic;
using System.Runtime.Serialization;


public static partial class UnityTFUtils
{
    // Mappings from Python calls to .NET
    static ObjectIDGenerator generator = new ObjectIDGenerator();

    public static long GetId(object x)
    {
        if (x == null)
            return 0;

        bool firstTime;
        return generator.GetId(x, out firstTime);
    }

    public static string ToString(object obj)
    {
        if (obj == null)
            return "null";

        if (obj is IEnumerable)
        {
            var l = new List<string>();
            foreach (object o in (IEnumerable)obj)
                l.Add(ToString(o));

            return "[" + string.Join(", ", l.ToArray()) + "]";
        }
        else if (obj is IDictionary)
        {
            var dict = obj as IDictionary;
            var l = new List<string>();
            foreach (object k in dict.Keys)
                l.Add($"{ToString(k)}: {ToString(dict[k])}");

            return "{" + string.Join(", ", l.ToArray()) + "}";
        }

        return obj.ToString();
    }
}