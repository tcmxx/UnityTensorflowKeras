using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.Runtime.Serialization;
using Accord.Math;
using static Current;


[DataContract]
public class MaxPooling2D : Layer
{
    private int[] poolSize;
    private int[] strides;
    private PaddingType padding;
    private DataFormatType? data_format;

    public MaxPooling2D(int[] pool_size, int[] strides, PaddingType padding, DataFormatType? data_format = null)
    {
        poolSize = pool_size.Copy();
        this.strides = strides.Copy();
        this.padding = padding;
        this.data_format = data_format;
    }


    protected override Tensor InnerCall(Tensor inputs, Tensor mask = null, bool? training = null)
    {
        return K.pool2D(inputs, poolSize, strides, padding, data_format, PoolMode.Max);
    }


    public override List<int?[]> compute_output_shape(List<int?[]> input_shapes)
    {
        if (input_shapes.Count != 1)
            throw new Exception("Expected a single input.");
        int?[] input_shape = input_shapes[0];

        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/layers/convolutional.py#L185

        if (this.data_format == DataFormatType.ChannelsLast)
        {
            var space = input_shape.Get(1, -1);
            var new_space = new List<int?>();
            for (int i = 0; i < space.Length; i++)
            {
                int? new_dim = ConvUtils.ConvOutputLength(
                    space[i],
                    this.poolSize[i],
                    padding: this.padding,
                    stride: this.strides[i]);
                new_space.Add(new_dim);
            }

            return new[] { new[] { input_shape[0] }.Concat(new_space).Concat(new int?[] { input_shape[space.Length] }).ToList().ToArray() }.ToList();
        }
        else if (this.data_format == DataFormatType.ChannelsFirst)
        {
            var space = input_shape.Get(2, 0);
            var new_space = new List<int?>();
            for (int i = 0; i < space.Length; i++)
            {
                int? new_dim = ConvUtils.ConvOutputLength(
                    space[i],
                    this.poolSize[i],
                    padding: this.padding,
                    stride: this.strides[i]);
                new_space.Add(new_dim);
            }

            return new[] { new[] { input_shape[0] }.Concat(new int?[] { input_shape[1] }).Concat(new_space).ToList().ToArray() }.ToList();
        }
        else
        {
            throw new Exception();
        }
    }
}