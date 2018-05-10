// Keras-Sharp: C# port of the Keras library

using System.Runtime.Serialization;
/// <summary>
///   He uniform variance scaling initializer.
/// </summary>
/// <remarks>
///  It draws samples from a uniform distribution within [-limit, limit] where <c>limit</c> is 
///  <c>sqrt(6 / fan_in)</c> where <c>fan_in</c> is the number of input units in the weight tensor.
/// </remarks>
/// 
[DataContract]
public class HeUniform : IWeightInitializer
{
    public int? seed;

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
    public UnityTFTensor Call(int[] shape, DataType? dtype = null)
    {
        return new VarianceScaling(scale: 2.0, mode: "fan_in", distribution: "uniform", seed: seed).Call(shape, dtype);
    }
}

