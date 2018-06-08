
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;
using UnityEngine;
using Accord.Math;
using System.IO;

public class UnityTFBackend : BackendBase, IBackend
{
    public TFGraph Graph { get; set; }

    public TFSession Session { get; set; }

    public readonly float EPS = 10e-8f;
    public readonly DataFormatType format = DataFormatType.ChannelsLast;

    // This dictionary holds a mapping {graph: learning_phase}.
    // A learning phase is a bool tensor used to run Keras models in
    // either train mode (learning_phase == 1) or test mode (learning_phase == 0).
    private Dictionary<TFGraph, object> _GRAPH_LEARNING_PHASES = new Dictionary<TFGraph, object>();

    // This dictionary holds a mapping {graph: UID_DICT}.
    // each UID_DICT is a dictionary mapping name prefixes to a current index,
    // used for generatic graph-specific string UIDs
    // for various names (e.g. layer names).
    private Dictionary<TFGraph, Dictionary<string, int>> _GRAPH_UID_DICTS = new Dictionary<TFGraph, Dictionary<string, int>>();


    public UnityTFBackend()
    {
        Graph = new TFGraph();
        Session = new TFSession(Graph);
    }

    /// <summary>
    ///   Get the uid for the default graph.
    /// </summary>
    /// 
    /// <param name="prefix">An optional prefix of the graph.</param>
    /// 
    /// <returns>A unique identifier for the graph.</returns>
    /// 
    public int get_uid(string prefix)
    {
        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L58
        var graph = Graph;
        if (!_GRAPH_UID_DICTS.ContainsKey(graph))
            _GRAPH_UID_DICTS[graph] = new Dictionary<string, int>();
        if (!_GRAPH_UID_DICTS[graph].ContainsKey(prefix))
            _GRAPH_UID_DICTS[graph][prefix] = 0;
        _GRAPH_UID_DICTS[graph][prefix] += 1;
        return _GRAPH_UID_DICTS[graph][prefix];
    }

    /// <summary>
    ///   Reset graph identifiers.
    /// </summary>
    /// 
    public void ResetUids()
    {
        _GRAPH_UID_DICTS = new Dictionary<TFGraph, Dictionary<string, int>>();
    }



    /// <summary>
    ///   Destroys the current TF graph and creates a new one.
    ///   Useful to avoid clutter from old models / layers.
    /// </summary>
    /// 
    public void clear_session()
    {
        Graph = new TFGraph();
        Session = new TFSession(Graph);

        TFOutput phase = Graph.Placeholder(dtype: TFDataType.Bool, operName: "keras_learning_phase");
        _GRAPH_LEARNING_PHASES = new Dictionary<TFGraph, object>();
        _GRAPH_LEARNING_PHASES[Graph] = phase;
    }


    public void ExportGraphDef(string filePath)
    {
        using (var buffer = new TFBuffer())
        {
            Graph.ToGraphDef(buffer);
            var bytes = buffer.ToArray();
            var fileInfo = new FileInfo(filePath);
            if (!Directory.Exists(fileInfo.Directory.FullName))
            {
                Directory.CreateDirectory(fileInfo.Directory.FullName);
            }
            
            File.WriteAllBytes(filePath, bytes);
        }
    }

    /// <summary>
    ///   Reshapes a tensor to the specified shape.
    /// </summary>
    /// 
    /// <param name="x">The Tensor or variable.</param>
    /// <param name="shape">The target shape.</param>
    /// 
    /// <returns>Tensor.</returns>
    /// 
    public Tensor reshape(Tensor x, int[] shape)
    {
        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L1724
        return Out(Graph.Reshape(tensor: In(x), shape: _constant(shape)));
    }



    public Tensor abs(Tensor input)
    {
        return Out(Graph.Abs(In(input)));
    }



    public Array get_value(Tensor x)
    {
        return x.eval() as Array;
    }


    public List<Array> batch_get_value(List<Tensor> weights)
    {
        List<Array> result = new List<Array>();
        foreach(var w in weights)
        {
            result.Add(w.eval() as Array);
        }
        //throw new NotImplementedException();
        return result;
    }
    public List<Array> batch_get_value(List<List<Tensor>> weights)
    {
        throw new NotImplementedException();
    }

    public void set_value(Tensor input, Array value) {
        UnityTFTensor x = (UnityTFTensor)input;
        //value = np.asarray(value, dtype = dtype(x))
        //var tf_dtype = tf.as_dtype(x.dtype.name.split('_')[0])
        var shape = new TFShape(x.TF_Shape);
        var valueTensor = UnityTFUtils.TFTensorFromArray(value, shape, 0,value.Length);

        TFOutput assignPlaceholder;
        TFOperation assignOperation;

        
        if (x.AssignPlaceHolder.HasValue) {
            assignPlaceholder = x.AssignPlaceHolder.Value;
            assignOperation = x.AssignOperation;
        }
        else {
            assignPlaceholder = Graph.Placeholder(valueTensor.TensorType, shape);
            assignOperation = Graph.Assign(x.Output, assignPlaceholder).Operation;//  x.assign(assign_placeholder)
            x.AssignPlaceHolder = assignPlaceholder;
            x.AssignOperation = assignOperation;
        }

        Session.Run(new TFOutput[] { x.AssignPlaceHolder.Value }, new TFTensor[] { valueTensor }, new TFOutput[] { }, new TFOperation[] { x.AssignOperation } );
     }

    public void batch_set_value(List<ValueTuple<Tensor, Array>> weight_value_tuples)
    {

        if (weight_value_tuples == null)
            return;

        var assignOps = new List<TFOperation>();
        var valueTensors = new List<TFTensor>();
        var assignPlaceholders = new List<TFOutput>();

        foreach (var p in weight_value_tuples)
        {
            var i1 = In(p.Item1);
            var shape = new TFShape(i1.TF_Shape);
            var valueTensor = UnityTFUtils.TFTensorFromArray(p.Item2, shape, 0, p.Item2.Length); //np.asarray(value, dtype = dtype(x))
            TFOutput assignPlaceholder;
            TFOperation assignOperation;

            if (i1.AssignPlaceHolder.HasValue)
            {
                assignPlaceholder = i1.AssignPlaceHolder.Value;
                assignOperation = i1.AssignOperation;
            }
            else
            {
                assignPlaceholder = Graph.Placeholder(valueTensor.TensorType, shape);
                assignOperation = Graph.Assign(i1.Output, assignPlaceholder).Operation;//  x.assign(assign_placeholder)
                i1.AssignPlaceHolder = assignPlaceholder;
                i1.AssignOperation = assignOperation;
            }
            assignOps.Add(assignOperation);
            valueTensors.Add(valueTensor);
            assignPlaceholders.Add(assignPlaceholder);
        }

        Session.Run(assignPlaceholders.ToArray(), valueTensors.ToArray(), new TFOutput[] { }, assignOps.ToArray());
        
    }
    

    /// <summary>
    ///   Binary crossentropy between an output tensor and a target tensor.
    /// </summary>
    /// 
    /// <param name="output">A tensor.</param>
    /// <param name="target">A tensor of the same shape as `output`.</param>
    /// <param name="from_logits">Whether `output` is expected to be a logits tensor. By default, we consider that `output` encodes a probability distribution.</param>
    /// 
    /// <returns>Output tensor.</returns>
    /// 
    public Tensor binary_crossentropy(Tensor output, Tensor target, bool from_logits = false)
    {
        //throw new NotImplementedException();
        TFOutput _output = In(output);
        TFOutput _target = In(target);
        TFDataType dtype = _output.OutputType;

        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L2792

        // Note: tf.nn.sigmoid_cross_entropy_with_logits
        // expects logits, Keras expects probabilities.
        if (!from_logits)
        {
            // transform back to logits
            TFOutput _epsilon = _constant(EPS, dtype: dtype);
            _output = Graph.Maximum(_output, _epsilon);
            _output = Graph.Minimum(_output, Graph.Sub(_constant(1, dtype: dtype), _epsilon));
            _output = Graph.Log(Graph.Div(_output, Graph.Sub(_constant(1, dtype: dtype), _output)));
        }

        return Out(Graph.SigmoidCrossEntropyWithLogits(labels: _target, logits: _output));
    }

    public Tensor cast(Tensor x, DataType dataType)
    {
        return Out(Graph.Cast(In(x), In(dataType)));
    }

    /// <summary>
    ///   Categorical crossentropy between an output tensor and a target tensor.
    /// </summary>
    /// 
    /// <param name="target">A tensor of the same shape as `output`.</param>
    /// <param name="output">A tensor resulting from a softmax (unless `from_logits` is True, in which case `output` is expected to be the logits).</param>
    /// <param name="from_logits">Boolean, whether `output` is the result of a softmax, or is a tensor of logits.</param>
    /// 
    /// <returns>Output tensor.</returns>
    /// 
    public Tensor categorical_crossentropy(Tensor target, Tensor output, bool from_logits = false)
    {
        var _target = In(target);
        var _output = In(output);

        // Note: tf.nn.softmax_cross_entropy_with_logits
        // expects logits, Keras expects probabilities.
        if (from_logits)
            _output.Output = Graph.Softmax(_output.Output);
        //if (!from_logits)
        //{
            // scale preds so that the class probas of each sample sum to 1
            int?[] shape = output.shape;
            var last = Graph.Const(new TFTensor(shape.Length - 1));
            TFOutput o = Graph.Div(_output, Graph.ReduceSum(_output, axis: last, keep_dims: true));
            // manual computation of crossentropy
            TFOutput _epsilon = _constant(EPS, dtype: _output.DType);

            o = Graph.Maximum(o, _epsilon);
            o = Graph.Minimum(o, Graph.Sub(_constant(1f), _epsilon));
            
            o = Graph.Neg(Graph.ReduceSum(Graph.Mul(_target, Graph.Log(_output)), axis: last));
            return Out(o);
        //}

        //return Out(ValueTuple.Create(Graph.so(_target, _output)).Item1);
        //throw new NotImplementedException();
    }

    public Tensor clip(Tensor norms, int v, int maxValue)
    {
        TFOutput o = Graph.Maximum(In(norms), _constant(v));
        o = Graph.Minimum(o, _constant(maxValue));
        return Out(o);
    }

    public Tensor clip(Tensor norms, double min_value, double max_value)
    {
        TFOutput o = Graph.Maximum(In(norms), _constant(min_value));
        o = Graph.Minimum(o, _constant(max_value));
        return Out(o);
    }
    public Tensor clip(Tensor norms, Tensor min_value, Tensor max_value)
    {
        TFOutput o = Graph.Maximum(In(norms), In(min_value));
        o = Graph.Minimum(o, In(max_value));
        return Out(o);
    }
    public Tensor clip_norm(Tensor g, double clipnorm, Tensor norm)
    {
        throw new NotImplementedException();
    }

    public Tensor constant<T>(T value, int[] shape = null, DataType? dtype = null, string name = null)
    {
        if (dtype == null)
            dtype = floatx();

        int[] _shape;
        if (shape == null)
        {
            Array arr = value as Array;
            if (arr != null)
                _shape = arr.GetLength();
            else _shape = new int[0];
            shape = _shape;
        }
        else
        {
            _shape = shape;
        }

        TFOutput o;
        if (shape != null && shape.Length != 0 && !(value is Array))
        {
           
            long length = shape.Aggregate((a, b) => (a * b));
            var tempValue = Vector.Create((int)length, value);
            //o = _constant(Matrix.Create(value.GetType(), _shape, value), In(dtype.Value), name);
            //o = _constant(Matrix.Create(value.GetType(), _shape), In(dtype.Value), name);
            //o = _constant(tempValue, Array.ConvertAll(shape, item => (long)item), In(dtype.Value),name);
            o = _constant(tempValue,  In(dtype.Value), name, Array.ConvertAll(shape, item => (long)item));
            //Debug.LogError("Currently the value neeed to be an array or shape length is not zero or no shape");
        }
        else
        {
            //Debug.Log("testset");
            if(shape != null && shape.Length != 0)
                o = _constant(value, In(dtype.Value), name, Array.ConvertAll(shape, item => (long)item));
            else
                o = _constant(value, In(dtype.Value), name);
            
        }

        //Debug.Log(string.Join(",", _int_shape(o)));
        if (!_int_shape(o).IsEqual(shape))
        {
            Debug.LogError("Shape:" + string.Join(",",_int_shape(o)) + ", and " + string.Join(",", shape) + " Not equal");
            //Debug.LogWarning("There might be a bug in the TensorflowSharp Graph.GetShape()/GetTensorShape(). Ignore it for now.");
            throw new Exception();
        }

        return Out(o);
    }


    private TFOutput _constant<T>(T value,  TFDataType? dtype = null, string operName = null, long[] shape = null)
    {
        //var temp = value as Array;
        //Debug.Log(value);
        //Debug.Log(temp.Length);
        TFTensor t = null;
        if (shape == null)
        {
            if(value is Array)
            {
                var dataArray = value as Array;
                t = new TFTensor(dataArray);
            }
            else
            {
                t = UnityTFUtils.TFTensorFromT(value);
            }
        }
        else
        {
            long length = shape.Aggregate((a, b) => (a * b));
            var dataArray = value as Array;
            Debug.Assert(dataArray != null, "Only support array input when shape is specified");
            t = UnityTFUtils.TFTensorFromArray(dataArray, new TFShape(shape), 0, (int)length);
            
        }
       // Debug.Log(string.Join(",",t.Shape));
        //var tensor = (float[])t.GetValue();
        //Debug.Log(string.Join(",", tensor));

        TFOutput o = Graph.Const(t, operName: operName);
        //TFStatus status = new TFStatus() ;
        //Debug.Log(string.Join(", ", Graph.GetShape(o.)));
        //Debug.Log(status.Ok);

        if (dtype == null || o.OutputType == dtype.Value)
            return o;

        return Graph.Cast(o, dtype.Value);
    }

    
    /*private TFOutput _constant<T>(T[] value, long[] shape, TFDataType? dtype = null, string operName = null)
    {
        long length = shape.Aggregate((a, b) => (a * b));
        Debug.Assert(length == value.Length, "Array size does not match shape");


        //TFTensor t = TFTensor.FromBuffer(new TFShape(shape), (dynamic)value,0, (int)length);
        TFTensor t = null;
        if(typeof(T) == typeof(float))
        {
            t = TFTensor.FromBuffer(new TFShape(shape), (float[]) Convert.ChangeType(value, typeof(float[])), 0, (int)length);
        }else if(typeof(T) == typeof(double))
        {
            t = TFTensor.FromBuffer(new TFShape(shape), (double[])Convert.ChangeType(value, typeof(float[])), 0, (int)length);
        }
        else
        {
            Debug.LogError("Does not Support Constant of type" + typeof(T).Name);
        }
        // Debug.Log(string.Join(",",t.Shape));
        //var tensor = (float[])t.GetValue();
        //Debug.Log(string.Join(",", tensor));

        TFOutput o = Graph.Const(t, operName: operName);
        //TFStatus status = new TFStatus() ;
        //Debug.Log(string.Join(", ", Graph.GetShape(o.)));
        //Debug.Log(status.Ok);

        if (dtype == null || o.OutputType == dtype.Value)
            return o;

        return Graph.Cast(o, dtype.Value);
    }*/


    public Tensor dropout(Tensor x, double keep_prob, int[] noise_shape, int? seed)
    {
        return Out(Graph.Dropout(In(x), _constant(keep_prob), new TFShape(noise_shape.Apply(p=>(long)p)),seed));
        //throw new NotImplementedException();
    }

    public DataType? dtype(Tensor tensor) => Out(In(tensor).DType);

    public Tensor elu(Tensor x)
    {
        return Out(Graph.Elu(In(x)));
    }

    public Tensor elu(Tensor x, double alpha)
    {
        throw new NotImplementedException();
    }

    public Tensor exp(Tensor x)
    {
        return Out(Graph.Exp(In(x)));
    }

    public Function function(List<Tensor> inputs, List<Tensor> outputs, List<List<Tensor>> updates, string name)
    {
        return new UnityTFFunction(this, inputs: inputs, outputs: outputs, updates: updates, name: name);
    }



    /// <summary>
    ///   Returns the shape of a variable.
    /// </summary>
    /// 
    public int?[] get_variable_shape(Tensor x)
    {
        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L2192
        return int_shape(x);
    }

    public List<Tensor> gradients(Tensor loss, List<Tensor> param)
    {
        var y = new TFOutput[] { In(loss).Output };
        var x = param.Select(t => In(t).Output).ToList().ToArray();

        TFOutput[] grads = Graph.AddGradients(y, x);
        List<Tensor> r = new List<Tensor>();
        for (int i = 0; i < grads.Length; i++)
            r.Add(Out(grads[i], name: "grad/" + x[i].Operation.Name));

        return r;
    }

    public Tensor greater_equal(Tensor w, double v)
    {
        throw new NotImplementedException();
    }

    public Tensor not_equal(Tensor x, Tensor y)
    {
        return Out(Graph.NotEqual(In(x), In(y)));
    }

    public Tensor not_equal<T>(Tensor x, T y) where T : struct
    {
        throw new NotImplementedException();
        /*using (this.name_scope("not_equal"))
        {
            UnityTFTensor _x = In(x);
            var _y = Graph.Cast(Graph.Const((dynamic)y), _x.DType);
            return Out(Graph.NotEqual(_x, _y));
        }*/
    }

    public Tensor hard_sigmoid(Tensor x)
    {
        throw new NotImplementedException();
    }

    public Tensor identity(Tensor x, string name = null)
    {
        return Out(Graph.Identity(In(x), operName: name));
    }

    /// <summary>
    ///   Returns the shape tensor or variable as a tuple of int or None entries.
    /// </summary>
    /// 
    /// <param name="x">Tensor or variable.</param>
    /// 
    /// <returns>A tuple of integers(or None entries).</returns>
    /// 
    public int?[] int_shape(Tensor x)
    {
        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L468

        if (x._keras_shape != null)
            return x._keras_shape;

        return _int_shape(In(x).Output);
    }

    private int?[] _int_shape(TFOutput _x)
    {
        try
        {
            //long[] shape = Graph.GetTensorShape(_x).ToArray();
            long[] shape = Graph.GetTensorShape(_x).ToArray() ;
            //Debug.Log(string.Join(",", Graph.GetTensorShape(_x)));
            return shape.Select(i => i == -1 ? null : (int?)i).ToList().ToArray();
        }
        catch
        {
            return null;
        }
    }

    public int?[] int_shape(TFTensor input_tensor)
    {
        throw new NotImplementedException();
    }

    /// <summary>
    ///   Selects `x` in train phase, and `alt` otherwise.
    /// </summary>
    /// 
    /// <param name="x">What to return in train phase.</param>
    /// <param name="alt">What to return otherwise.</param>
    /// <param name="training">Optional scalar tensor specifying the learning phase.</param>
    /// 
    /// <returns>Either 'x' or 'alt' based on the 'training' flag. The 'training' flag defaults to 'K.learning_phase()'.</returns>
    /// 
    public Tensor in_train_phase(Func<Tensor> x, Func<Tensor> alt, bool? training)
    {
        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L2583

        bool uses_learning_phase;

        if (training == null)
        {
            var t = learning_phase();
            if (t is bool)
                training = (bool)t;
            uses_learning_phase = true;
        }
        else
        {
            uses_learning_phase = false;
        }

        if (training == true)
        {
            return x();
        }
        else if (training == false)
        {
            return alt();
        }
        else
        {
            //else: assume learning phase is a placeholder tensor.

            Tensor xx = @switch((Tensor)learning_phase(), x, alt);

            if (uses_learning_phase)
                xx._uses_learning_phase = true;
            return xx;
        }
    }

    /// <summary>
    ///   Selects `x` in test phase, and `alt` otherwise. Note that `alt` should have the* same shape* as `x`.
    /// </summary>
    public Tensor in_test_phase(Func<Tensor> x, Func<Tensor> alt, bool? training = null)
    {
        return in_train_phase(alt, x, training: training);
    }
    

    /// <summary>
    ///   Switches between two operations depending on a scalar value. Note that both `then_expression` and `else_expression`
    ///   should be symbolic tensors of the *same shape
    /// </summary>
    /// 
    /// <param name="condition">The condition: scalar tensor(`int` or `bool`).</param>
    /// <param name="then_expression">Either a tensor, or a callable that returns a tensor.</param>
    /// <param name="else_expression">Either a tensor, or a callable that returns a tensor.</param>
    /// 
    /// <returns>The selected tensor.</returns>
    /// 
    public Tensor @switch(Tensor condition, Func<Tensor> then_expression, Func<Tensor> else_expression)
    {
        var _condition = In(condition);

        if (_condition.DType != TFDataType.Bool)
            condition = Out(Graph.Cast(_condition, TFDataType.Bool));
        TFOutput x = Graph.Cond(In(condition),
                    () => In(then_expression()),
                    () => In(else_expression()));
        return Out(x);
    }

    public bool is_sparse(Tensor tensor)
    {
        return false;
    }

    public Tensor l2_normalize(Tensor expected, int axis)
    {
        throw new NotImplementedException();
    }

    /// <summary>
    ///   Returns the learning phase flag.
    /// </summary>
    /// 
    /// <remarks>
    ///   The learning phase flag is a bool tensor(0 = test, 1 = train)
    ///   to be passed as input to any Keras function
    ///   that uses a different behavior at train time and test time.
    /// </remarks>
    /// 
    /// <returns> Learning phase (scalar integer tensor or Python integer).</returns>
    /// 
    public object learning_phase()
    {
        TFGraph graph = Graph;
        if (!_GRAPH_LEARNING_PHASES.ContainsKey(graph))
        {
            TFOutput phase = Graph.Placeholder(dtype: TFDataType.Bool, operName: "keras_learning_phase");
            _GRAPH_LEARNING_PHASES[graph] = phase;
        }

        return _GRAPH_LEARNING_PHASES[graph];
    }

    /// <summary>
    ///   Sets the learning phase to a fixed value.
    /// </summary>
    public void set_learning_phase(bool value)
    {
        _GRAPH_LEARNING_PHASES[Graph] = value;
    }

    public Tensor max(Tensor x, int v, object p)
    {
        throw new NotImplementedException();
    }

    public Tensor max(Tensor x, int axis, bool keepdims)
    {
        var o = Graph.Max(In(x), Graph.Const(axis), keepdims);
        return Out(o);
    }

    public Tensor max(Tensor tensor, int axis)
    {
        var o = Graph.Max(In(tensor), Graph.Const(axis), false);
        return Out(o);
    }

    public Tensor min(Tensor a, Tensor b)
    {
        var o = Graph.Minimum(In(a), In(b));
        return Out(o);
    }
    public Tensor min(Tensor x, int axis, bool keepdims)
    {
       var o = Graph.Min(In(x), Graph.Const(axis), keepdims);
        return Out(o);
    }


    public Tensor maximum(double v, Tensor tensor)
    {
        throw new NotImplementedException();
    }

    /// <summary>
    ///   Turn a nD tensor into a 2D tensor with same 0th dimension. In other words, it flattens each data samples of a batch.
    /// </summary>
    /// 
    public Tensor batch_flatten(Tensor x)
    {
        //throw new NotImplementedException();
        var _x = In(x);
        TFOutput shape = Graph.Shape(_x);
        TFOutput dim = Graph.Prod(Graph.Slice(shape, Graph.Const(1), Graph.Rank(shape)), reduction_indices: Graph.ReduceDims(shape, null));
        return Out(Graph.Reshape(In(x), Graph.Stack(new TFOutput[] { Graph.Const(-1), dim })));
    }


    public TFOutput normalize_axis(int[] axis, int? ndim)
    {
        axis = (int[])axis.Clone();
        for (int i = 0; i < axis.Length; i++)
        {
            if (axis[i] < 0)
                axis[i] = axis[i] % ndim.Value;
        }

        return Graph.Const(axis);
    }

    /// <summary>
    ///   Mean of a tensor, alongside the specified axis.
    /// </summary>
    /// 
    /// <param name="x">A tensor or variable.</param>
    /// <param name="axis">A list of integer. Axes to compute the mean.</param>
    /// <param name="keepdims>A boolean, whether to keep the dimensions or not. If <paramref name="keepdims"/> is <c>false</c>, 
    ///   the rank of the tensor is reduced by 1 for each entry in <paramref name="axis"/>. If <paramref name="keepdims"/> is 
    ///   <c>true</c>, the reduced dimensions are retained with length 1.
    ///   
    /// <returns>A tensor with the mean of elements of <c>x</c>.</returns>
    /// 
    public Tensor mean(Tensor x, int[] axis, bool keepdims = false, string name = null)
    {
        return Out(Graph.Mean(In(x), normalize_axis(axis, ndim(x)), keepdims, operName: name));
    }
    /// <summary>
    ///   Mean of a tensor, alongside the specified axis.
    /// </summary>
    /// 
    /// <param name="x">A tensor or variable.</param>
    /// <param name="axis">The axis where to compute the mean.</param>
    /// <param name="keepdims>A boolean, whether to keep the dimensions or not. If <paramref name="keepdims"/> is <c>false</c>, 
    ///   the rank of the tensor is reduced by 1 for each entry in <paramref name="axis"/>. If <paramref name="keepdims"/> is 
    ///   <c>true</c>, the reduced dimensions are retained with length 1.
    ///   
    /// <returns>A tensor with the mean of elements of <c>x</c>.</returns>
    /// 
    public Tensor mean(Tensor x, int axis = -1, bool keepdims = false, string name = null)
    {
        return Out(Graph.Mean(In(x), reduction_indices: Graph.Const(axis), keep_dims: keepdims, operName: name));
        //return Out(Graph.Mean(x.Output, reduction_indices: Graph.Const(0)));
    }


    public Tensor minus(Tensor tensor)
    {
        throw new NotImplementedException();
    }

    public Tensor dot(Tensor a, Tensor b, string name = null)
    {
        return Out(Graph.MatMul(In(a).Output, In(b).Output, operName: name));
    }


    public Tensor mul<T>(T a, Tensor b, string name = null)
    {
        return mul(constant(a, dtype: dtype(b)), b, name: name);
    }

    public Tensor mul(Tensor a, Tensor b, string name = null)
    {
        
        return Out(Graph.Mul(In(a).Output, In(b).Output, operName: name));
    }

    public Tensor mul<T>(Tensor a, T b, string name = null)
    {
        return mul(a, constant(b, dtype: dtype(a), name: name));
    }

    public Tensor mul(List<Tensor> batch_outs, int length)
    {
        throw new NotImplementedException();
    }





    public Tensor div<T>(T a, Tensor b)
    {
        return div(constant(a, dtype: dtype(b)), b);
    }

    public Tensor div(Tensor a, Tensor b)
    {
        return Out(Graph.Div(In(a).Output, In(b).Output));
    }

    public Tensor div<T>(Tensor a, T b)
    {
        return div(a, constant(b, dtype: dtype(a)));
    }

    public Tensor div(List<Tensor> batch_outs, int length)
    {
        throw new NotImplementedException();
    }



    public Tensor add(Tensor a, Tensor b)
    {
        return Out(Graph.Add(In(a).Output, In(b).Output));
    }

    public Tensor bias_add(Tensor a, Tensor b, DataFormatType? data_format = null, string name = null)
    {
        return Out(Graph.BiasAdd(In(a), In(b), data_format: In(data_format), operName: name));
    }

    private string In(DataFormatType? data_format)
    {
        if (data_format == null)
            return null;

        switch (data_format.Value)
        {
            case DataFormatType.ChannelsFirst:
                return "NHWC";
            case DataFormatType.ChannelsLast:
                return "NHWC";
            default:
                throw new Exception();
        }
    }

    public Tensor add<T>(T a, Tensor b)
    {
        return add(constant(a), b);
    }

    public Tensor add<T>(Tensor a, T b)
    {
        return add(a, constant(b));
    }



    public Tensor subtract(Tensor a, Tensor b, string name = null)
    {
        return Out(Graph.Sub(In(a).Output, In(b).Output, operName: name));
    }

    public Tensor subtract<T>(T a, Tensor b, string name = null)
    {
        return subtract(constant(a), b, name: name);
    }

    public Tensor subtract<T>(Tensor a, T b, string name = null)
    {
        return subtract(a, constant(b), name: name);
    }



    public NameScope name_scope(string name)
    {
        return new TensorFlowNameScope(Graph.WithScope(name), name);
    }

    public NameScope name_scope(string operName, string userName)
    {
        string name = MakeName(operName, userName);
        return new TensorFlowNameScope(Graph.WithScope(name), name);
    }



    /// <summary>
    /// Returns the number of axes in a tensor, as an integer.
    /// </summary>
    /// <param name="x">Tensor or variable.</param>
    /// <example>
    /// <codesrc="TensorFlowBackendTest.cs" region="doc_ndim">
    /// </example>
    /// 
    public int? ndim(Tensor x)
    {
        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L519

        int?[] dims = x.shape;

        if (dims != null)
            return dims.Length;

        return Graph.GetTensorNumDims(In(x).Output);
    }

    public Tensor placeholder(int?[] shape = null, int? ndim = null, DataType? dtype = null, bool sparse = false, string name = null)
    {
        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L397

        if (sparse)
            throw new NotImplementedException();

        if (dtype == null)
            dtype = floatx();

        if (shape == null)
        {
            if (ndim != null)
                shape = new int?[ndim.Value];
        }

        var tfshape = this.In(shape);
        Tensor x = Out(Graph.Placeholder(In(dtype.Value), tfshape, operName: name));
        x._keras_shape = shape;
        x._uses_learning_phase = false;
        return x;
    }

    /// <summary>
    ///   Returns a tensor with uniform distribution of values.
    /// </summary>
    /// <param name="shape">A tuple of integers, the shape of tensor to create.</param>
    /// <param name="minval">A float, lower boundary of the uniform distribution to draw samples.</param>
    /// <param name="maxval">A float, upper boundary of the uniform distribution to draw samples.</param>
    /// <param name="dtype">The dtype of returned tensor.</param>
    /// <param name="seed">The random seed.</param>
    /// 
    /// <returns>A tensor.</returns>
    /// 
    public Tensor random_uniform(int[] shape, double minval = 0.0, double maxval = 1.0, DataType? dtype = null, int? seed = null, string name = null)
    {
        if (dtype == null)
            dtype = floatx();

        var _dtype = In(dtype.Value);

        if (seed == null)
            seed = Accord.Math.Random.Generator.Random.Next(1000000);

        using (name_scope("random_uniform",name ))
        {
            
               var _shape = Graph.Const(shape.Select(x => (long)x).ToList().ToArray());
            TFOutput u = Graph.RandomUniform(_shape, dtype: _dtype, seed: seed, operName: "uniform");

            return Out(Graph.Add(Graph.Mul(u, _constant(maxval - minval, dtype: _dtype)),
                                        _constant(minval, dtype: _dtype)), name: "scaled");
        }
    }

    public Tensor relu(Tensor x)
    {
        return Out(Graph.Relu(In(x)));
    }

    public Tensor sigmoid(Tensor x)
    {
        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L2817
        return Out(Graph.Sigmoid(In(x)));
    }

    public Tensor softmax(Tensor x)
    {
        return Out(Graph.Softmax(In(x).Output));
    }

    public Tensor softplus(Tensor x)
    {
        throw new NotImplementedException();
    }

    public Tensor softsign(Tensor x)
    {
        throw new NotImplementedException();
    }

    public Tensor sqrt(Tensor x)
    {
        return Out(Graph.Sqrt(In(x)));
    }

    public Tensor pow(Tensor x, Tensor p, string name = null)
    {
        return Out(Graph.Pow(In(x), In(p), operName: name));
    }

    public Tensor square(Tensor w)
    {
        return Out(Graph.Square(In(w)));
    }

    public Tensor sum(Tensor x, int[] axis, bool keepdims = false, string name = null)
    {
        return Out(Graph.ReduceSum(In(x), Graph.Const(axis), keepdims, name));
    }

    public Tensor sum(Tensor x, int axis, bool keepdims = false, string name = null)
    {
        return Out(Graph.ReduceSum(In(x), Graph.Const(axis), keepdims, name));
    }

    public Tensor sum(List<Tensor> x, int[] axis = null, bool keepdims = false, string name = null)
    {
        throw new NotImplementedException();
        
    }

    public Tensor sum(Tensor tensor)
    {
        throw new NotImplementedException();
    }

    public Tensor sum(double v, Tensor tensor)
    {
        throw new NotImplementedException();
    }

    public Tensor tanh(Tensor x)
    {
        return Out(Graph.Tanh(In(x)));
    }

    public Tensor truncated_normal(int[] shape, double v, double stddev, DataType? dtype, int? seed)
    {
        throw new NotImplementedException();
    }

    public Tensor truncated_normal(int?[] shape, double v, double stddev, DataType? dtype, int? seed)
    {
        throw new NotImplementedException();
    }


    public Tensor update(Tensor x, Tensor new_x, string name = null)
    {
        UnityTFTensor _x = In(x);
        //var result = new UnityTFTensor(this);
        //result.Operation = Graph.AssignVariableOp(_x.Output, In(new_x), operName: name);
        //return result;
        return Out(Graph.Assign(_x.Output, In(new_x), operName: name));
    }
    /// <summary>
    /// Temperaly output TFOperation right now
    /// </summary>
    public Tensor update_add<T>(Tensor x, T increment, string name = null)
        where T : struct
    {
        UnityTFTensor _x = In(x);
        //var result = new UnityTFTensor(this);
        //result.Operation = Graph.AssignAddVariableOp(_x, _constant(increment), operName: name);
        //return result;
        return Out(Graph.AssignAdd(_x, _constant(increment), operName: name));
    }

    public Tensor print_tensor(Tensor x, string message)
    {
        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L2204
        UnityTFTensor _x = In(x);
        return Out(Graph.Print(_x, new[] { _x.Output }, message));
    }

    /// <summary>
    ///   Instantiates a variable and returns it.
    /// </summary>
    /// 
    /// <param name="value">C# array, initial value of the tensor.</param>
    /// <param name="dtype">Tensor type.</param>
    /// <param name="name">Optional name string for the tensor.</param>
    /// 
    public Tensor variable<T>(T value, DataType? dtype = null, string name = null)
        where T : struct
    {
        if (dtype == null)
            dtype = floatx();

        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L308
        var _dtype = In(dtype.Value);

        using (var scope = name_scope("Variable", name))
        {
            var t = new UnityTFTensor(this);
            
            

            //t.Output = Graph.Variable(init,operName: "var");
            t.Output = Graph.VariableV2(TFShape.Scalar, _dtype, operName: "var");
            var init = _constant(value, _dtype, operName: "init");
            init = Graph.Print(init, new[] { init }, $"initializing {scope.Name}");
            
            Graph.AddInitVariable(Graph.Assign(t.Output, init, operName: "assign").Operation);

            t._keras_shape = new int?[] { };
            t._uses_learning_phase = false;
            return t;
        }
    }

    /// <summary>
    ///   Instantiates a variable and returns it.
    /// </summary>
    /// 
    /// <param name="array">C# array, initial value of the tensor.</param>
    /// <param name="dtype">Tensor type.</param>
    /// <param name="name">Optional name string for the tensor.</param>
    /// 
    public Tensor variable(Array array, DataType? dtype = null, string name = null)
    {
        if (dtype == null)
            dtype = floatx();

        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L308
        var _dtype = In(dtype.Value);

        var t = new UnityTFTensor(this);
        t.Output = Graph.VariableV2(In(array.GetLength()), _dtype, operName: name);

        

        //t.Output = Graph.Variable(init, operName: name);
        string varName = t.Output.Operation.Name;

        var init = _constant(array, _dtype, operName: $"{(name != null?name:"Variable"+UnityTFUtils.GetId(t.Output).ToString())}/init");
        init = Graph.Print(init, new[] { init }, $"initializing {varName}");
        //Graph.AddInitVariable(Graph.AssignVariableOp(t.Output, init, operName: $"{varName}/assign"));
        Graph.AddInitVariable(Graph.Assign(t.Output, init, operName: $"{varName}/assign").Operation);

        t._keras_shape = array.GetLength().Apply(x => (int?)x);
        t._uses_learning_phase = false;
        return t;
    }

    /// <summary>
    ///   Instantiates a variable and returns it.
    /// </summary>
    /// 
    /// <param name="tensor">Tensor, initial value of the tensor.</param>
    /// <param name="dtype">Tensor type.</param>
    /// <param name="name">Optional name string for the tensor.</param>
    /// 
    public Tensor variable(Tensor tensor, DataType? dtype = null, string name = null)
    {
        if (dtype == null)
            dtype = floatx();

        var _tensor = In(tensor);
        var _dtype = In(dtype.Value);
        TFShape _shape = In(tensor.shape);

        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L308

        var t = new UnityTFTensor(this);

        t.Output = Graph.VariableV2(_shape, _dtype, operName: name);

        string varName = t.Output.Operation.Name;

        TFOutput init;
        if (_tensor.TensorTF == null)
            init = _tensor.Output;
        else
            init = Graph.Cast(Graph.Const(_tensor.TensorTF), _dtype, operName: $"{name}/init");

        //Debug.Log("tensorshape:" + string.Join(",",_tensor.Tensor.Shape));
        //TFOperation initOp;
        //TFOutput value;
        //t.Output = Graph.VariableV2(new TFShape(_tensor.Tensor.Shape), _dtype);
        //t.Output = Graph.Variable(init, out initOp, out value, operName: name);
        //string varName = t.Output.Operation.Name;


        init = Graph.Print(init, new[] { init }, $"initializing {varName}");
        //Graph.AddInitVariable(Graph.AssignVariableOp(t.Output, init, operName: $"{varName}/assign"));
        Graph.AddInitVariable(Graph.Assign(t.Output, init, operName: $"{varName}/assign").Operation);
        t._keras_shape = tensor.shape;
        t._uses_learning_phase = false;
        return t;
    }

    public Tensor transpose(Tensor tensor)
    {
        return Out(Graph.Transpose(In(tensor).Output));
    }

    public Tensor transpose(Tensor tensor, int[] perm)
    {
        return Out(Graph.Transpose(In(tensor).Output, _constant(perm)));
    }


    public object eval(Tensor tensor)
    {
        var _tensor = In(tensor);
        if (_tensor.ValueOnly)
        {
            return _tensor.TensorTF.GetValue();
        }
        else
        {
            return eval(_tensor.Output);
        }
    }

    public object eval(TFOutput output)
    {
        try
        {
            // Initialize variables if necessary
            TFOperation[] ops = Graph.GetGlobalVariablesInitializer();
            if (ops.Length > 0)
                Session.Run(new TFOutput[] { }, new TFTensor[] { }, new TFOutput[] { }, ops);
        }
        catch
        {
            // temporary workaround until changes are sent to TensorFlowSharp
        }

        // Evaluate tensor
        TFTensor[] result = Session.Run(new TFOutput[] { }, new TFTensor[] { }, new[] { output });

        if (result.Length == 1)
            return result[0].GetValue();

        return result.Apply(x => x.GetValue());
    }



    public Tensor conv1d(Tensor inputs, Tensor kernel, int strides, PaddingType padding, DataFormatType? data_format = null, int dilation_rate = 1, string name = null)
    {
        throw new NotImplementedException();
    }

    public Tensor conv2d(Tensor inputs, Tensor kernel, int[] strides, PaddingType padding, DataFormatType? data_format = null, int[] dilation_rate = null, string name = null)
    {
        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L3102
        if (data_format == null)
            data_format = image_data_format();

        if (dilation_rate == null)
            dilation_rate = new int[] { 1, 1 };

        if (!dilation_rate.IsEqual(new[] { 1, 1 }))
            throw new NotImplementedException();

        TFOutput x = In(inputs).Output;
        TFOutput _kernel = In(kernel).Output;

        var _strides = new List<int>(strides); _strides.Insert(0, 1); _strides.Add(1);

        // With 4d inputs, tf.nn.convolution only supports
        // data_format NHWC, so we transpose the inputs
        // in case we are in data_format channels_first.
        x = _preprocess_conv2d_input(x, data_format.Value);
        string _padding = _preprocess_padding(padding);
        x = Graph.Conv2D(
            input: x,
            filter: _kernel,
            //dilation_rate: dilation_rate,
            strides: _strides.ToArray().Select(i => (long)i).ToList().ToArray(),
            padding: _padding,
            data_format: "NHWC");
        return Out(_postprocess_conv2d_output(x, data_format.Value));
    }

    public Tensor pool2D(Tensor input, int[] poolSize, int[] strides,
           PaddingType padding, DataFormatType? dataFormat = null,
           PoolMode poolMode= PoolMode.Max)
    {
        if (dataFormat == null)
            dataFormat = image_data_format();
        string _padding = _preprocess_padding(padding);
        var _strides = new List<int>(strides); _strides.Insert(0, 1); _strides.Add(1);
        var _poolSize = new List<int>(poolSize); _poolSize.Insert(0, 1); _poolSize.Add(1);

        TFOutput x = In(input).Output;

        var o = _preprocess_conv2d_input(x, dataFormat.Value);

        if (poolMode == PoolMode.Max)
        {
            o = Graph.MaxPool(x, Array.ConvertAll(_poolSize.ToArray(), item => (long)item), Array.ConvertAll(_strides.ToArray(), item => (long)item), _padding);
        }
        else if (poolMode == PoolMode.Average)
        {
            o = Graph.AvgPool(x, Array.ConvertAll(_poolSize.ToArray(), item => (long)item), Array.ConvertAll(_strides.ToArray(), item => (long)item), _padding);
        }
        else
            Debug.LogError("Invalid pooling mode");

        return Out(_postprocess_conv2d_output(o, dataFormat.Value));
     }


    /// <summary>
    ///   Transpose and cast the output from conv2d if needed.
    /// </summary>
private TFOutput _postprocess_conv2d_output(TFOutput x, DataFormatType data_format)
    {
        if (data_format == DataFormatType.ChannelsFirst)
            x = Graph.Transpose(x, _constant(new[] { 0, 3, 1, 2 }));

        if (floatx() == DataType.Double)
            x = Graph.Cast(x, TFDataType.Double);
        return x;
    }

    /// <summary>
    ///   Convert keras' padding to tensorflow's padding.
    /// </summary>
    /// 
    public string _preprocess_padding(PaddingType padding)
    {
        switch (padding)
        {
            case PaddingType.Same:
                return "SAME";
            case PaddingType.Valid:
                return "VALID";
        }

        throw new ArgumentException($"Invalid padding: {padding}");
    }

    /// <summary>
    ///   Transpose and cast the input before the conv2d.
    /// </summary>
    private TFOutput _preprocess_conv2d_input(TFOutput x, DataFormatType data_format)
    {
        if (x.OutputType == TFDataType.Double)
            x = Graph.Cast(x, TFDataType.Float);

        if (data_format == DataFormatType.ChannelsFirst)
        {
            // TF uses the last dimension as channel dimension,
            // instead of the 2nd one.
            // TH input shape: (samples, input_depth, rows, cols)
            // TF input shape: (samples, rows, cols, input_depth)
            x = Graph.Transpose(x, _constant(new[] { 0, 2, 3, 1 }));
        }

        return x;
    }

    public Tensor conv3d(Tensor inputs, Tensor kernel, int[] strides, PaddingType padding, DataFormatType? data_format = null, int[] dilation_rate = null, string name = null)
    {
        throw new NotImplementedException();
    }



    /// <summary>
    ///   Instantiates an all-zeros variable and returns it.
    /// </summary>
    /// <param name="shape">Tuple of integers, shape of returned Keras variable.</param>
    /// <param name="dtype">Data type of returned Keras variable.</param>
    /// <param name="name">String, name of returned Keras variable.</param>
    /// <returns>A variable(including Keras metadata), filled with <c>0.0</c>.</returns>
    public Tensor zeros(int?[] shape, DataType? dtype = null, string name = null)
    {
        return zeros(shape.Select(i => i.Value).ToList().ToArray(), dtype, name);
    }

    /// <summary>
    ///   Instantiates an all-zeros variable and returns it.
    /// </summary>
    /// <param name="shape">Tuple of integers, shape of returned Keras variable.</param>
    /// <param name="dtype">Data type of returned Keras variable.</param>
    /// <param name="name">String, name of returned Keras variable.</param>
    /// <returns>A variable(including Keras metadata), filled with <c>0.0</c>.</returns>
    public Tensor zeros(int[] shape, DataType? dtype = null, string name = null)
    {
        if (dtype == null)
            dtype = floatx();

        // The following is not necessary since C# is strongly typed:
        // shape = tuple(map(int, shape))
        // tf_dtype = _convert_string_dtype(dtype)

        // However, we might have to perform other conversions of our own:
        //Type type = TFTensor.TypeFromTensorType(In(dtype.Value));
        Type type = dtype.Value.ToType();
        Array zeros = Array.CreateInstance(type, shape);

        return this.variable(array: zeros, name: name);
    }

    /// <summary>
    ///   Element-wise equality between two tensors.
    /// </summary>
    /// 
    /// <param name="x">Tensor or variable.</param>
    /// <param name="y">Tensor or variable.</param>
    /// 
    /// <returns>A bool tensor.</returns>
    /// 
    public Tensor equal(Tensor x, Tensor y)
    {
        return Out(Graph.Equal(In(x), In(y)));
    }

    /// <summary>
    ///   Returns the index of the maximum value along an axis.
    /// </summary>
    /// 
    /// <param name="x">Tensor or variable.</param>
    /// <param name="axis">The axis along which to perform the reduction.</param>
    /// 
    /// <returns>A tensor.</returns>
    /// 
    public Tensor argmax(Tensor x, int axis = -1)
    {
        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/backend/tensorflow_backend.py#L1332
        //axis = _normalize_axis(axis, ndim(x));
        return Out(Graph.ArgMax(In(x), Graph.Const(axis)));
    }

    public Tensor round(Tensor x)
    {
        return Out(Graph.Round(In(x)));
    }




public DataType floatx()
    {
        return DataType.Float;
    }


    public string MakeName(string operName, string userName)
    {
        if (userName == null)
        {
            var k = Graph.CurrentNameScope == "" ? operName : Graph.CurrentNameScope + "/" + operName;
            return $"{k}_{UnityTFUtils.ToString(get_uid(k))}";
        }

        if (Graph.CurrentNameScope == "")
            return userName;
        return Graph.CurrentNameScope + "/" + userName;
    }



    #region conversion

    public TFShape In(int?[] shape)
    {
        return new TFShape(shape.Select(x => x.HasValue ? (long)x.Value : -1).ToList().ToArray());
    }
    public UnityTFTensor In(Tensor output)
    {
        return (UnityTFTensor)output;
    }
    public TFShape In(int[] shape)
    {
        return new TFShape(shape.Select(x => (long)x).ToList().ToArray());
    }

    public Tensor Out(TFOutput output, string name = null)
    {
        if (name != null)
            output = Graph.Identity(output, operName: name);

        return new UnityTFTensor(this)
        {
            Output = output
        };
    }

    public UnityTFTensor In(TFOutput output)
    {
        return new UnityTFTensor(this) { Output = output };
    }

    public static TFDataType In(DataType dataType)
    {
        return (TFDataType)dataType;
    }

    public static TFDataType? In(DataType? dataType)
    {
        if (dataType == null)
            return null;
        return (TFDataType)dataType.Value;
    }

    public static DataType? Out(TFDataType? dataType)
    {
        if (dataType == null)
            return null;
        return Out(dataType.Value);
    }

    public static DataType Out(TFDataType dataType)
    {
        if ((int)dataType > 100)
            return (DataType)((dataType - 100));
        return (DataType)dataType;
    }

    #endregion





    #region IDisposable Support
    private bool disposedValue = false; // To detect redundant calls

    protected virtual void Dispose(bool disposing)
    {
        if (!disposedValue)
        {
            if (disposing)
            {
                // TODO: dispose managed state (managed objects).
                if (Graph != null)
                    Graph.Dispose();
                if (Session != null)
                    Session.Dispose();
            }

            // TODO: free unmanaged resources (unmanaged objects) and override a finalizer below.
            // TODO: set large fields to null.

            disposedValue = true;
            Graph = null;
            Session = null;
        }
    }

    // TODO: override a finalizer only if Dispose(bool disposing) above has code to free unmanaged resources.
    // ~TensorFlowBackend() {
    //   // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
    //   Dispose(false);
    // }

    // This code added to correctly implement the disposable pattern.
    public void Dispose()
    {
        // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
        Dispose(true);
        // TODO: uncomment the following line if the finalizer is overridden above.
        // GC.SuppressFinalize(this);
    }
    #endregion
}