using static Current;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

/// <summary>
/// <summary>
///   Scaled Exponential Linear Unit (SeLU) activation function.
/// </summary>
/// 
/// <seealso cref="KerasSharp.IActivationFunction" />
/// 
[DataContract]
public class ELU : ActivationFunctionBase, IActivationFunction
{


    public override Tensor Call(Tensor x, Tensor mask = null)
    {
        return K.elu(x);
    }
}
