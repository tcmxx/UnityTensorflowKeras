
using System;
using System.Threading;
using System.Reflection;

public static class Current
{
    private static ThreadLocal<UnityTFBackend> backend;

    private static string[] assemblyNames =
    {
            "UnityTFBackend",
        };

    public static string Name = "UnityTFBackend";

    public static UnityTFBackend K
    {
        get { return backend.Value; }
        set { backend.Value = value; }
    }

    static Current()
    {
        backend = new ThreadLocal<UnityTFBackend>(() => load(Name));
    }

    public static void Switch<T>()
    {
        Switch(typeof(T).FullName);
    }

    public static void Switch(string backendName)
    {
        Name = backendName;
        backend.Value = load(Name);
    }



    private static UnityTFBackend load(string typeName)
    {
        //Type type = find(typeName);
        Type type = typeof(UnityTFBackend);
        UnityTFBackend obj = (UnityTFBackend)Activator.CreateInstance(type);

        return obj;
    }

    private static Type find(string typeName)
    {
        foreach (string assemblyName in assemblyNames)
        {
            try
            {
                Assembly assembly = Assembly.Load(assemblyName);

                var types = assembly.GetExportedTypes();

                foreach (var type in types)
                {
                    string currentTypeName = type.FullName;
                    if (currentTypeName == typeName)
                        return type;
                }
            }
            catch
            {
                // TODO: Remove this try-catch block by a proper check
            }
        }

        throw new ArgumentException("typeName");
    }

}

