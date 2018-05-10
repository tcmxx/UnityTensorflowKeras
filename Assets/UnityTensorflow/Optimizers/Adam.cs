
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;

using static Current;

[DataContract]
public class Adam : OptimizerBase, IOptimizer
{
    private UnityTFTensor iterations;
    private UnityTFTensor lr;
    private UnityTFTensor beta_1;
    private UnityTFTensor beta_2;
    private UnityTFTensor decay;
    private double initial_decay;
    private double epsilon;

    public Adam(double lr = 0.001, double beta_1 = 0.9, double beta_2 = 0.999, double epsilon = 1e-8, double decay = 0.0)
    {
        this.iterations = K.Variable(0f, name: "iterations");
        this.lr = K.Variable(lr, name: "lr");
        this.beta_1 = K.Variable(beta_1, name: "beta_1");
        this.beta_2 = K.Variable(beta_2, name: "beta_2");
        this.epsilon = epsilon;
        this.decay = K.Variable(decay, name: "decay");
        this.initial_decay = decay;
    }

    public List<List<UnityTFTensor>> get_updates(List<UnityTFTensor> param, Dictionary<UnityTFTensor, IWeightConstraint> constraints, UnityTFTensor loss)
    {
        using (K.NameScope($"adam"))
        {
            var grads = this.get_gradients(loss, param);
            this.updates = new List<List<UnityTFTensor>> { new List<UnityTFTensor> { K.UpdateAdd(this.iterations, 1f, "iterations/update") } };

            UnityTFTensor lr = this.lr;
            if (this.initial_decay > 0)
                lr *= (1 / (1 + this.decay * this.iterations));

            UnityTFTensor t = this.iterations + 1;
            UnityTFTensor lr_t = K.Mul(lr, (K.Sqrt(1 - K.Pow(this.beta_2, t)) /
                             (1 - K.Pow(this.beta_1, t))), name: "lr_t");

            var shapes = param.Select(p => K.get_variable_shape(p));
            var ms = shapes.Select(shape => K.Zeros(shape)).ToArray();
            var vs = shapes.Select(shape => K.Zeros(shape)).ToArray();
            this.weights = new[] { this.iterations }.Concat(ms).Concat(vs).ToList();

            for (int i = 0; i < param.Count; i++)
            {
                using (K.NameScope($"{param[i].Name}"))
                {
                    var p = param[i];
                    var g = grads[i];
                    var m = ms[i];
                    var v = vs[i];
                    var m_t = (this.beta_1 * m) + (1 - this.beta_1) * g;
                    var v_t = (this.beta_2 * v) + (1 - this.beta_2) * K.Square(g);
                    var p_t = K.Subtract(p, lr_t * m_t / (K.Sqrt(v_t) + this.epsilon), name: "p_t");

                    this.updates.Add(new List<UnityTFTensor> { K.Update(m, m_t, "m_t/update") });
                    this.updates.Add(new List<UnityTFTensor> { K.Update(v, v_t, "v_t/update") });

                    var new_p = p_t;
                    // apply constraints
                    if (constraints.Keys.Contains(p))
                    {
                        var c = constraints[p];
                        new_p = c.Call(new_p);
                    }

                    this.updates.Add(new List<UnityTFTensor> { K.Update(p, new_p, "parameter/update") });
                }
            }

            return this.updates;
        }
    }
}
