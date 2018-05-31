
using System.Runtime.Serialization;

/// <summary>
///   Glorot uniform initializer, also called Xavier uniform initializer.
/// </summary>
/// 
/// <remarks>
///  It draws samples from a uniform distribution within [-limit, limit] where <c>limit</c> is 
///  <c>sqrt(6 / fan_in)</c> where <c>fan_in</c> is the number of input units in the weight tensor.
/// </remarks>
/// 
[DataContract]
public class GlorotUniform : IWeightInitializer
{
    private int? seed;

    /// <summary>
    /// Initializes a new instance of the <see cref="GlorotUniform"/> class.
    /// </summary>
    /// 
    /// <param name="seed">The integer used to seed the random generator.</param>
    /// 
    public GlorotUniform(int? seed = null)
    {
        this.seed = seed;
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
        return new VarianceScaling(scale: 1.0, mode: "fan_avg", distribution: "uniform", seed: seed).Call(shape, dtype);
    }
}

