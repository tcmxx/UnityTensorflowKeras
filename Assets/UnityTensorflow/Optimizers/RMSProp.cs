
using System;
using System.Collections.Generic;
using System.Runtime.Serialization;

using static Current;
using System.Linq;

/// <summary>
///   RMSProp optimizer.
/// </summary>
/// 
/// <remarks>
///   It is recommended to leave the parameters of this optimizer at their default values (except the learning rate, which can be freely tuned).
/// </remarks>
/// 
[DataContract]
public class RMSProp : OptimizerBase, IOptimizer
{
    private UnityTFTensor decay;
    private double initial_decay;
    private UnityTFTensor iterations;
    private UnityTFTensor lr;
    private UnityTFTensor rho;
    private double epsilon;

    // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/optimizers.py#L190

    public RMSProp()
        : this(lr: 0.001, rho: 0.9, epsilon: 1e-8, decay: 0.0)
    {

    }

    public RMSProp(double lr, double rho = 0.9, double epsilon = 1e-8, double decay = 0.0)
    {
        this.lr = K.Variable(lr, name: "lr");
        this.rho = K.Variable(rho, name: "rho");
        this.epsilon = epsilon;
        this.decay = K.Variable(decay, name: "decay");
        this.initial_decay = decay;
        this.iterations = K.Variable(0.0, name: "iterations");
    }

    public List<List<UnityTFTensor>> get_updates(List<UnityTFTensor> param, Dictionary<UnityTFTensor, IWeightConstraint> constraints, UnityTFTensor loss)
    {
        using (K.NameScope($"rmsprop"))
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/optimizers.py#L221

            List<UnityTFTensor> grads = this.get_gradients(loss, param);
            List<int?[]> shapes = param.Select(p => K.get_variable_shape(p)).ToList();
            List<UnityTFTensor> accumulators = shapes.Select(shape => K.Zeros(shape)).ToList();
            this.weights = accumulators;
            this.updates = new List<List<UnityTFTensor>>();

            UnityTFTensor lr = this.lr;
            if (this.initial_decay > 0)
            {
                lr = lr * (1.0 / (1.0 + this.decay * this.iterations));
                this.updates.Add(new List<UnityTFTensor> { K.UpdateAdd(this.iterations, 1) });
            }

            for (int i = 0; i < param.Count; i++)
            {
                using (K.NameScope($"{param[i].Name}"))
                {
                    UnityTFTensor p = param[i];
                    UnityTFTensor g = grads[i];
                    UnityTFTensor a = accumulators[i];

                    // update accumulator
                    UnityTFTensor new_a = this.rho * a + (1.0 - this.rho) * K.Square(g);
                    this.updates.Add(new List<UnityTFTensor> { K.Update(a, new_a) });
                    UnityTFTensor new_p = p - lr * g / (K.Sqrt(new_a) + this.epsilon);

                    // apply constraints
                    if (constraints.ContainsKey(p))
                    {
                        IWeightConstraint c = constraints[p];
                        new_p = c.Call(new_p);
                    }

                    this.updates.Add(new List<UnityTFTensor> { K.Update(p, new_p) });
                }
            }

            return this.updates;
        }
    }
}
