
using System.Runtime.Serialization;
using static Current;


/// <summary>
///   Initializer that generates tensors initialized to a constant value.
/// </summary>
/// 
[DataContract]
public class Constant : IWeightInitializer
{
    private double value;

    /// <summary>
    /// Initializes a new instance of the <see cref="Constant"/> class.
    /// </summary>
    /// <param name="value">The value.</param>
    public Constant(double value)
    {
        this.value = value;
    }

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
        return K.constant(value, shape: shape, dtype: dtype);
    }
}

