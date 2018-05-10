
using System;
using System.Runtime.Serialization;

[DataContract]
public class CustomLoss : ILoss
{
    // TODO: Use a delegate instead...
    Func<UnityTFTensor, UnityTFTensor, UnityTFTensor, UnityTFTensor, UnityTFTensor> loss;

    public CustomLoss(Func<UnityTFTensor, UnityTFTensor, UnityTFTensor, UnityTFTensor, UnityTFTensor> loss)
    {
        this.loss = loss;
    }

    public UnityTFTensor Call(UnityTFTensor expected, UnityTFTensor actual, UnityTFTensor sample_weight = null, UnityTFTensor mask = null)
    {
        return loss(expected, actual, sample_weight, mask);
    }
}
