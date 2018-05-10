
using System;
using TensorFlow;

public class TensorFlowNameScope : NameScope, IDisposable
{
    string name;
    TFScope scope;

    public override string Name { get { return name; } }

    public TensorFlowNameScope(TFScope scope, string name)
    {
        this.scope = scope;
        this.name = name;
    }

    public override void Dispose()
    {
        scope.Dispose();
    }
}