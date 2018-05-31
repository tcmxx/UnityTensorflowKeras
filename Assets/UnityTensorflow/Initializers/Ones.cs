
using System.Runtime.Serialization;
using static Current;

/// <summary>
///   Initializer that generates tensors initialized to 1.
/// </summary>
/// 
[DataContract]
public class Ones : IWeightInitializer
{

    /// <summary>
    /// Creates a <see cref="TFTensor" /> with the desired initial weights.
    /// </summary>
    /// 
    /// <param name="shape">The shape of the tensor to be generated.</param>
    /// <param name="dtype">The <see cref="TFDataType">data type</see> of the tensor to be generated.</param>
    /// <returns>A <see cref="TFTensor" /> initialized of dimensions <paramref name="shape" />
    /// and element data type <paramref name="dtype" /> that has been initialized using this
    /// strategy.</returns>
    /// 
    public Tensor Call(int[] shape, DataType? dtype = null)
    {
        return K.Constant(1, shape: shape, dtype: dtype);
    }
}

