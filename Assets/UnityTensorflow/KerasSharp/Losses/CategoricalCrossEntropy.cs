

using System;
using System.Runtime.Serialization;


using static Current;

[DataContract]
public class CategoricalCrossEntropy : ILoss
{

    /// <summary>
    ///   Wires the given ground-truth and predictions through the desired loss.
    /// </summary>
    /// 
    /// <param name="expected">The ground-truth data that the model was supposed to approximate.</param>
    /// <param name="actual">The actual data predicted by the model.</param>
    /// 
    /// <returns>A scalar value representing how far the model's predictions were from the ground-truth.</returns>
    /// 
    public Tensor Call(Tensor expected, Tensor actual, Tensor sample_weight = null, Tensor mask = null)
    {
        if (sample_weight != null || mask != null)
            throw new NotImplementedException();

        using (K.name_scope("categorical_crossentropy"))
            return K.categorical_crossentropy(expected, actual);
    }
}
