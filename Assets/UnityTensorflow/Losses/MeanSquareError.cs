
using System;
using static Current;

public class MeanSquareError : ILoss
{
    public MeanSquareError()
    {
    }

    public Tensor Call(Tensor y_true, Tensor y_pred, Tensor sample_weight = null, Tensor mask = null)
    {
        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/losses.py#L7

        if (sample_weight != null || mask != null)
            throw new NotImplementedException();

        using (K.name_scope("mean_square_error"))
        {

            return K.mean(K.square(y_pred - y_true), axis: -1);
        }
    }
}
