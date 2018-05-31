
using System;
using System.Runtime.Serialization;

[DataContract]
public class CustomLoss : ILoss
{
    // TODO: Use a delegate instead...
    Func<Tensor, Tensor, Tensor, Tensor, Tensor> loss;

    public CustomLoss(Func<Tensor, Tensor, Tensor, Tensor, Tensor> loss)
    {
        this.loss = loss;
    }

    public Tensor Call(Tensor expected, Tensor actual, Tensor sample_weight = null, Tensor mask = null)
    {
        return loss(expected, actual, sample_weight, mask);
    }
}
