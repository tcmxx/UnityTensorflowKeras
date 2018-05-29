using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TensorFlow;
using System;

public class UnityTFTensor {

    public TFTensor Tensor { get; set; }
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

    public UnityTFBackend Backend { get; set; }
    public TFDataType DType
    {
        get
        {
            if (Tensor != null)
                return Tensor.TensorType;
            return Output.OutputType;
        }
    }

    public DataType? dtype
    {
        get { return Backend.DType(this); }
    }

    public bool UsesLearningPhase { get; set; }

    public long[] TF_Shape
    {
        get
        {
            var tf = Backend.Graph;
            return tf.GetShape(Output);
        }
    }

    public string Name
    {
        get { return Output.Operation.Name; }
    }

    public int?[] Shape
    {
        get { return Backend.IntShape(this); }
    }


    public int?[] KerasShape
    {
        get; set;
    } = null;
    public int?[] IntShape
    {
        get;set;
    } = null;

    public ValueTuple<Layer, int , int >? KerasHistory    //layer node_index, tensor_index
    {
        get;
        set;
    }

    public TFOutput? AssignPlaceHolder { get; set; } = null;
    public TFOperation AssignOperation { get; set; } = null;




    public UnityTFTensor(UnityTFBackend backend)
    {
        this.Backend = backend;
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



    public object Eval()
    {
        return Backend.Eval(this);
    }



    public static UnityTFTensor operator *(double a, UnityTFTensor b)
    {
        return b.Backend.Mul(a, b);
    }

    public static UnityTFTensor operator *(UnityTFTensor a, int b)
    {
        return a.Backend.Mul(a, b);
    }

    public static UnityTFTensor operator *(int a, UnityTFTensor b)
    {
        return b.Backend.Mul(a, b);
    }

    public static UnityTFTensor operator *(UnityTFTensor a, UnityTFTensor b)
    {
        return b.Backend.Mul(a, b);
    }

    public static UnityTFTensor operator /(double a, UnityTFTensor b)
    {
        return b.Backend.Div(a, b);
    }

    public static UnityTFTensor operator /(UnityTFTensor a, int b)
    {
        return a.Backend.Div(a, b);
    }

    public static UnityTFTensor operator /(int a, UnityTFTensor b)
    {
        return b.Backend.Div(a, b);
    }

    public static UnityTFTensor operator /(UnityTFTensor a, UnityTFTensor b)
    {
        return b.Backend.Div(a, b);
    }

    public static UnityTFTensor operator +(double a, UnityTFTensor b)
    {
        return b.Backend.Add(a, b);
    }

    public static UnityTFTensor operator +(int a, UnityTFTensor b)
    {
        return b.Backend.Add(a, b);
    }

    public static UnityTFTensor operator +(UnityTFTensor a, UnityTFTensor b)
    {
        return b.Backend.Add(a, b);
    }

    public static UnityTFTensor operator +(UnityTFTensor a, double b)
    {
        return a.Backend.Add(a, b);
    }

    public static UnityTFTensor operator -(double a, UnityTFTensor b)
    {
        return b.Backend.Subtract(a, b);
    }

    public static UnityTFTensor operator -(UnityTFTensor a, UnityTFTensor b)
    {
        return b.Backend.Subtract(a, b);
    }
}
