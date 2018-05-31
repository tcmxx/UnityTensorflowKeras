
using Accord.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;

using static Current;


/// <summary>
///   Abstract optimizer base class.
/// </summary>
/// 
/// <seealso cref="KerasSharp.Models.IOptimizer" />
/// 
[DataContract]
public abstract class OptimizerBase
{
    protected List<List<Tensor>> updates;
    protected List<Tensor> weights;
    public double clipnorm;
    public double clipvalue;

    protected OptimizerBase()
    {
        var allowed_kwargs = new[] { "clipnorm", "clipvalue" };

        //foreach (var k in kwargs)
        //    if (!allowed_kwards.Contains(k))
        //        throw new Exception("Unexpected keyword argument passed to optimizer: " + k);
        // this.__dict__.update(kwargs)

        this.updates = new List<List<Tensor>>();
        this.weights = new List<Tensor>();
    }

    public virtual void get_updates(object param, IWeightConstraint constraints, ILoss loss)
    {
        throw new NotImplementedException();
    }

    public List<Tensor> get_gradients(Tensor loss, List<Tensor> param)
    {
        List<Tensor> grads = K.gradients(loss, param);

        if (this.clipnorm > 0)
        {
            var norm = K.sqrt(K.sum(grads.Select(g => K.sum(K.square(g))).ToList()));
            grads = grads.Select(g => K.clip_norm(g, this.clipnorm, norm)).ToList();
        }

        if (clipvalue > 0)
            grads = grads.Select(g => K.clip(g, -this.clipvalue, this.clipvalue)).ToList();

        return grads;
    }

    /// <summary>
    ///   Sets the weights of the optimizer, from Numpy arrays.
    /// </summary>
    /// 
    /// <remarks>
    ///  Should only be called after computing the gradients (otherwise the optimizer has no weights).
    /// </remarks>
    /// 
    /// <param name="weights">The list of Numpy arrays. The number of arrays and their shape must match 
    ///   number of the dimensions of the weights of the optimizer(i.e.it should match the output of 
    ///   <see cref="get_weights"/></param>
    /// 
    public void set_weights(List<Array> weights)
    {
        var param = this.weights;
        var weight_value_tuples = new List<ValueTuple<Tensor, Array>>();
        var param_values = K.batch_get_value(param);

        for (int i = 0; i < param_values.Count; i++)
        {
            Array pv = param_values[i];
            Tensor p = param[i];
            Array w = weights[i];

            if (pv.GetLength().IsEqual(w.GetLength()))
                throw new Exception($"Optimizer weight shape {pv.GetLength()} not compatible with provided weight shape {w.GetLength()}.");

            weight_value_tuples.Add(ValueTuple.Create(p, w));
        }

        K.batch_set_value(weight_value_tuples);
    }

    /// <summary>
    ///   Returns the current value of the weights of the optimizer.
    /// </summary>
    public List<Array> get_weights()
    {
        return K.batch_get_value(this.weights);
    }
}
