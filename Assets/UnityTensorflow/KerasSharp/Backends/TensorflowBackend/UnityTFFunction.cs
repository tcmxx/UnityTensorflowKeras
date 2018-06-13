
using System;
using System.Collections.Generic;
using TensorFlow;
using System.Linq;
using Accord.Math;
using Accord;
using System.IO;
using UnityEngine;

public class UnityTFFunction: Function
{
    private UnityTFBackend backend;
    private TFGraph graph;
    private List<Tensor> inputs;
    private List<Tensor> outputs;
    private string name;
    private List<TFOperation> updates_op;

    public UnityTFFunction(UnityTFBackend k, List<Tensor> inputs, List<Tensor> outputs, List<List<Tensor>> updates, string name)
    {
        this.backend = k;
        this.graph = k.Graph;

        if (updates == null)
            updates = new List<List<Tensor>>();
        this.inputs = inputs;
        this.outputs = outputs;
        {
            var updates_ops = new List<TFOperation>();
            foreach (List<Tensor> update in updates)
            {
                if (update.Count == 2)
                {
                    var p = backend.In(update[0]);
                    var new_p = backend.In(update[1]);
                    updates_ops.Add(graph.AssignVariableOp(p, new_p));
                }
                else if (update.Count == 1)
                {
                    // assumed already an op
                    updates_ops.Add(backend.In(update[0]).Output.Operation);
                }
                else
                {
                    throw new NotSupportedException();
                }
            }

            //this.updates_op = tf.group(updates_ops);
            this.updates_op = updates_ops;
        }

        this.name = name;
        //this.session_kwargs = session_kwargs;
    }

    public override List<Tensor> Call(List<Array> inputs)
    {
        var feed_dict = new Dictionary<Tensor, Array>();
        foreach (var tuple in Enumerable.Zip(this.inputs, inputs, (a, b) => Tuple.Create(a, b)))
        {
            // if (is_sparse(tensor))
            // {
            //     sparse_coo = value.tocoo()
            //     indices = np.concatenate((np.expand_dims(sparse_coo.row, 1),
            //                               np.expand_dims(sparse_coo.col, 1)), 1)
            //     value = (indices, sparse_coo.data, sparse_coo.shape)
            // }
            feed_dict[tuple.Item1] = tuple.Item2;
        }

        var session = backend.Session;

        var init = graph.GetGlobalVariablesInitializer();
        if (init.Length > 0)
        {
            Console.WriteLine("Initializing variables:");
            foreach (var op in init)
            {
                Console.WriteLine(" - " + op.Name);
                session.Run(new TFOutput[0], new TFTensor[0], new TFOutput[0], new[] { op });
            }

            Console.WriteLine("Operations:");
            foreach (var op in graph.GetEnumerator())
                Console.WriteLine(" - " + op.Name);
            Console.WriteLine();
        }

        //Console.WriteLine("Before:");
        //PrintVariables(feed_dict, session);
        // Console.ReadKey();

        var runner = session.GetRunner();

        foreach (var o in this.outputs)
            runner.Fetch(backend.In(o).Output);

        foreach (var op in this.updates_op)
            runner.AddTarget(op);


        List<TFTensor> tensors = new List<TFTensor>();
        foreach (KeyValuePair<Tensor, Array> pair in feed_dict)
        {
            UnityTFTensor t = backend.In(pair.Key);

            //get the shape based on the tensor and input data length
            long[] actualShape = t.TF_Shape.Copy();
            int totalLength = Mathf.Abs((int)actualShape.Aggregate((s, n) => n * s));

            int indexOfBatch = actualShape.IndexOf(-1);
            if (indexOfBatch >= 0)
                actualShape[indexOfBatch] = pair.Value.Length / totalLength;
            Debug.Assert(totalLength <= pair.Value.Length, "Feed array does not have enough data");

            //Debug.Log("totalLength:"+totalLength + "  Shape:" + string.Join(",", actualShape));

            //TFTensor data = TFTensor.FromBuffer(new TFShape(actualShape), (dynamic)pair.Value, 0, totalLength *(pair.Value.Length / totalLength));
            TFTensor data = UnityTFUtils.TFTensorFromArray(pair.Value, new TFShape(actualShape));

            tensors.Add(data);
            runner.AddInput(t.Output, data);
        }



        var updated = runner.Run();

        foreach(var d in tensors)
        {
            d.Dispose();
        }
        //Console.WriteLine();

        //foreach (var v in updated)
        //{
        //    object obj = v.GetValue();
        //    if (obj is float[,])
        //        Console.WriteLine((obj as float[,]).ToCSharp());
        //    else if (obj is float[])
        //        Console.WriteLine((obj as float[]).ToCSharp());
        //    else
        //        Console.WriteLine(obj);
        //}

        //Console.WriteLine();
        //Console.WriteLine();

        //Console.WriteLine("After:");
        //PrintVariables(feed_dict, session);

        return updated.Get(0, this.outputs.Count).Select(t => {
            var result = new UnityTFTensor(backend);
            result.TensorValue = t.GetValue();
            result.TensorType = t.TensorType;
            return (Tensor)result;
        }).ToList();

        // Console.ReadKey();
    }

    private void PrintVariables(Dictionary<Tensor, Array> feed_dict, TFSession session)
    {
        string[] ops =
        {
                //"SGD/grad/dense_1/dense_1/kernel/var",
                //"SGD/grad/dense_2/dense_2/kernel/var",
                //"SGD/grad/dense_2/dense_2/bias/var",
                //"loss/dense_1_loss/y_true",
                //"loss/dense_1_loss/y_pred",
                //"loss/dense_1_loss/weights",
                //"iterations/var",
                //"lr/var",
                //"lr_t",
                //"p_t",
                //"metrics/binary_accuracy/Round0",
                //"metrics/binary_accuracy/Cast0",
                //"metrics/binary_accuracy/Mean0",
                //"metrics/binary_accuracy/Equal0",
                //"metrics/binary_accuracy/value",
                //"metrics/score_array/mean"
                //"beta_1/var",
                //"beta_2/var",
                //"decay/var",
                //"adam/grad/dense_1/dense_1/kernel/var",
                //"dense_1/variance_scaling/1/scaled",
                //"dense_1/dense_1/kernel/var",
                //"dense_1/call/dot",
                //"dense_1/call/Sigmoid0",
            };

        foreach (var op in ops)
        {
            try
            {
                var debugRunner = session.GetRunner();
                foreach (KeyValuePair<Tensor, Array> pair in feed_dict)
                {
                    UnityTFTensor t = backend.In(pair.Key);
                    debugRunner.AddInput(t.Output, pair.Value);
                }

                Console.WriteLine(op);
                debugRunner.Fetch(op);

                var v = debugRunner.Run();

                object obj = v[0].GetValue();

                foreach(var va in v)
                {
                    va.Dispose();
                }
                if (obj is float[,])
                    Console.WriteLine((obj as float[,]).ToCSharp());
                else if (obj is float[])
                    Console.WriteLine((obj as float[]).ToCSharp());
                else if (obj is bool[,])
                    Console.WriteLine((obj as bool[,]).ToCSharp());
                else if (obj is bool[])
                    Console.WriteLine((obj as bool[]).ToCSharp());
                else if (obj is sbyte[,])
                    Console.WriteLine((obj as sbyte[,]).ToCSharp());
                else if (obj is sbyte[])
                    Console.WriteLine((obj as sbyte[]).ToCSharp());
                else
                    Console.WriteLine(obj);
            }
            catch
            {

            }
        }
    }
}
