
//
//This is modified from KerasSharp repo for use of Unity., by Xiaoxiao Ma, Aalto University, 
//
// Keras-Sharp: C# port of the Keras library
// https://github.com/cesarsouza/keras-sharp
//
// Based under the Keras library for Python. See LICENSE text for more details.
//
//    The MIT License(MIT)
//    
//    Permission is hereby granted, free of charge, to any person obtaining a copy
//    of this software and associated documentation files (the "Software"), to deal
//    in the Software without restriction, including without limitation the rights
//    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//    copies of the Software, and to permit persons to whom the Software is
//    furnished to do so, subject to the following conditions:
//    
//    The above copyright notice and this permission notice shall be included in all
//    copies or substantial portions of the Software.
//    
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//    SOFTWARE.
//
namespace KerasSharp.Engine.Topology
{
    using UnityEngine;
    using TensorFlow;
    using KerasSharp.Backends;

    public class UnityTFTensor : Tensor
    {

        public object TensorValue { get; set; }
        public TFDataType TensorType { get; set; }
        public TFOutput Output
        {
            get
            {
                Debug.Assert(!ValueOnly, "This Tensor is value only and does not have a graph");
                Debug.Assert(output.HasValue, "This Tensor does not have  TFOutput");
                return output.Value;
            }
            set
            {
                output = value;
            }
        }
        private TFOutput? output = null;

        public bool ValueOnly
        {
            get
            {
                return !output.HasValue && TensorValue != null;
            }
        }

        public TFOperation Operation
        {
            get
            {
                if (operation != null)
                {
                    return operation;
                }
                else
                {
                    return Output.Operation;
                }
            }
            set
            {
                operation = value;
            }
        }
        private TFOperation operation = null;

        public new UnityTFBackend K
        {
            get { return base.K as UnityTFBackend; }
        }


        public TFDataType DType
        {
            get
            {
                if (TensorValue != null)
                    return TensorType;
                return Output.OutputType;
            }
        }


        public long[] TF_Shape
        {
            get
            {
                var tf = K.Graph;
                return tf.GetShape(Output);
            }
        }
        public override string name
        {
            get { return Output.Operation.Name; }
        }
        public TFOutput? AssignPlaceHolder { get; set; } = null;
        public TFOperation AssignOperation { get; set; } = null;


        public UnityTFTensor(IBackend backend)
        : base(backend)
        {
        }


        public override string ToString()
        {
            string n = Output.Operation.Name;
            long i = Output.Index;
            string s = string.Join(", ", TF_Shape);
            string r = $"UnityTFTensor '{n}_{i}' shape={s} dtype={DType}";
            return r;
        }

        public static implicit operator TFOutput(UnityTFTensor t)
        {
            return t.Output;
        }

    }
}