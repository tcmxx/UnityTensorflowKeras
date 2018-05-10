
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;

using static Current;


/// <summary>
///   Stochastic gradient descent optimizer.
/// </summary>
/// 
/// <remarks>
///  Includes support for momentum, learning rate decay, and Nesterov momentum.
/// </remarks>
/// 
/// <seealso cref="KerasSharp.Models.IOptimizer" />
/// 
[DataContract]
public class SGD : OptimizerBase, IOptimizer
{
    private UnityTFTensor iterations;
    private UnityTFTensor lr;
    private UnityTFTensor momentum;
    private UnityTFTensor decay;
    private double initial_decay;
    private bool nesterov;

    public SGD()
        : this(0.01, 0.0, 0.0, false)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SGD" /> class.
    /// </summary>
    /// 
    /// <param name="lr">float >= 0. Learning rate.</param>
    /// <param name="momentum">float >= 0. Parameter updates momentum.</param>
    /// <param name="decay">float >= 0. Learning rate decay over each update.</param>
    /// <param name="nesterov">Whether to apply Nesterov momentum.</param>
    /// 
    public SGD(double lr = 0.01, double momentum = 0.0, double decay = 0.0, bool nesterov = false)
        : base()
    {
        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/optimizers.py#L144

        this.iterations = K.Variable(0, name: "iterations");
        this.lr = K.Variable(lr, name: "lr");
        this.momentum = K.Variable(momentum, name: "momentum");
        this.decay = K.Variable(decay, name: "decay");
        this.initial_decay = decay;
        this.nesterov = nesterov;
    }

    public List<List<UnityTFTensor>> get_updates(List<UnityTFTensor> param, Dictionary<UnityTFTensor, IWeightConstraint> constraints, UnityTFTensor loss)
    {
        using (K.NameScope($"SGD"))
        {
            var grads = this.get_gradients(loss, param);
            this.updates = new List<List<UnityTFTensor>>();

            if (this.initial_decay > 0)
                this.lr *= (1 / (1 + this.decay * this.iterations));

            this.updates.Add(new List<UnityTFTensor> { K.UpdateAdd(this.iterations, 1f, name: "iterations/update") });

            // momentum
            List<UnityTFTensor> moments;

            using (K.NameScope("moments"))
            {
                List<int?[]> shapes = param.Select(p => K.get_variable_shape(p)).ToList();
                moments = shapes.Select(s => K.Zeros(s)).ToList();
            }

            this.weights = new[] { this.iterations }.Concat(moments).ToList();

            for (int i = 0; i < param.Count; i++)
            {
                using (K.NameScope($"{param[i].Name}"))
                {
                    UnityTFTensor p = param[i];
                    UnityTFTensor g = grads[i];
                    UnityTFTensor m = moments[i];

                    UnityTFTensor v = this.momentum * m - lr * g;  // velocity

                    this.updates.Add(new List<UnityTFTensor> { K.Update(m, v, "momentum/update") });

                    UnityTFTensor new_p;
                    if (this.nesterov)
                        new_p = p + this.momentum * v - lr * g;
                    else
                        new_p = p + v;

                    // apply constraints
                    if (constraints.ContainsKey(p))
                        new_p = constraints[p].Call(new_p);

                    updates.Add(new List<UnityTFTensor> { K.Update(p, new_p, "parameter/update") });
                }
            }

            return this.updates;
        }
    }

}
