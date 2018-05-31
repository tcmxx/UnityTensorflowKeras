using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TensorFlow;
using System;
using System.Linq;
using Accord;

public class UnityTFTensor: Tensor
{

    public TFTensor TensorTF { get; set; }
    public TFOutput Output { get; set; }
    public TFOperation Operation { get
        {
            if (operation != null)
            {
                return operation;
            }
            else
            {
                return Output.Operation;
            }
        }
        set{
            operation = value;
        } }
    private TFOperation operation = null;

    public new UnityTFBackend K
    {
        get { return base.K as UnityTFBackend; }
    }


    public TFDataType DType
    {
        get
        {
            if (TensorTF != null)
                return TensorTF.TensorType;
            return Output.OutputType;
        }
    }


    public long[] TF_Shape
    {
        get
        {
            var tf = K.Graph;
            return tf.GetShape(Output);
        }
    }
    public override string name
    {
        get { return Output.Operation.Name; }
    }
    public TFOutput? AssignPlaceHolder { get; set; } = null;
    public TFOperation AssignOperation { get; set; } = null;


    public UnityTFTensor(IBackend backend)
    : base(backend)
    {
    }


    public override string ToString()
    {
        string n = Output.Operation.Name;
        long i = Output.Index;
        string s = string.Join(", ", TF_Shape);
        string r = $"UnityTFTensor '{n}_{i}' shape={s} dtype={DType}";
        return r;
    }

    public static implicit operator TFOutput(UnityTFTensor t)
    {
        return t.Output;
    }

   
}
