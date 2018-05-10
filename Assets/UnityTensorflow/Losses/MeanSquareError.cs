
    using System;
    using static Current;

    public class MeanSquareError : ILoss
    {
        public MeanSquareError()
        {
        }

        public UnityTFTensor Call(UnityTFTensor y_true, UnityTFTensor y_pred, UnityTFTensor sample_weight = null, UnityTFTensor mask = null)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/losses.py#L7

            if (sample_weight != null || mask != null)
                throw new NotImplementedException();

        using (K.NameScope("mean_square_error"))
            
            return K.Mean(K.Square(y_pred - y_true), axis: -1);
        }
    }
