using CommunityToolkit.HighPerformance;
using CommunityToolkit.HighPerformance.Buffers;
using cslm;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace cslm
{
    public enum DType
    {
        dt_f32,
        dt_f16,
        dt_bf16,
        dt_f8e5m2,
        dt_f8e4m3,
        dt_i32,
        dt_i16,
        dt_i8,
        dt_u8,
    };

    public struct Tensor
    {
        public string name_;
        public DType dtype_;
        public int shape0_;
        public int shape1_;
        public int shape2_;
        public int shape3_;
        public long data_;
        public long size_;
    };

    public struct Metadata
    {
        public string key_;
        public long value_;
    }

    public class Tensors
    {
        private long size_;
        private byte[]? data_;
        private List<Metadata> metadata_ = new List<Metadata>();
        private List<Tensor> tensors_ = new List<Tensor>();

        public Tensor this[int index]
        {
            get
            {
                return tensors_[index];
            }
        }

        public static async Task<Tensors> OpenAsync(string path)
        {
            try
            {
                byte[] bytes = await System.IO.File.ReadAllBytesAsync(path);
                Tensors tensors = Tensors.JsonParser.parse(bytes);
                return tensors;
            }catch(Exception e)
            {
                Debug.WriteLine(e.Message);
                return null;
            }
        }

        public int find(string name, int layer)
        {
            name = string.Format(name, layer);
            for(int i=0; i<tensors_.Count; ++i)
            {
                if (tensors_[i].name_ == name)
                {
                    return i;
                }
            }
            return -1;
        }

        public int find(string name, int layer, DType dtype, int shape0, int shape1, int shape2, int shape3)
        {
            int index = find(name, layer);
            if (index < 0)
            {
                return -1;
            }
            if (tensors_[index].dtype_ != dtype
                || tensors_[index].shape0_ != shape0
                || tensors_[index].shape1_ != shape1
                || tensors_[index].shape2_ != shape2
                || tensors_[index].shape3_ != shape3)
            {
                return -1;
            }
            return index;
        }

        public int find_metadata(string key)
        {
            for (int i = 0; i < metadata_.Count; ++i)
            {
                if(key == metadata_[i].key_)
                {
                    return i;
                }
            }
            return -1;
        }

        public string get_metadata_value(int index)
        {
            long offset = metadata_[index].value_;
            int length = 0;
            while (data_[offset] != 0)
            {
                ++length;
                ++offset;
            }
            return Encoding.ASCII.GetString(data_, (int)offset, length);
        }

        private class JsonParser
        {
            private const int BufferSize = 1024;
            private byte[] buffer_ = new byte[BufferSize];

            public long json_skipws(long offset, byte[] data)
            {
                while (offset < data.LongLength && (data[offset] == ' ' || data[offset] == '\t' || data[offset] == '\n' || data[offset] == '\r'))
                {
                    ++offset;
                }
                return offset;
            }

            public long json_string(long offset, byte[] data, out long result)
            {
                result = -1;
                if (data[offset] != '"')
                {
                    return data.LongLength;
                }
                ++offset;
                result = offset;
                while (data[offset] != '"')
                {
                    if (data[offset] == 0 || data[offset] == '\\')
                    {
                        return data.LongLength;
                    }
                    ++offset;
                }
                data[offset] = 0;
                return json_skipws(offset+1, data);
            }

            public long json_string(long offset, byte[] data, out string result)
            {
                result = string.Empty;
                long str;
                offset = json_string(offset, data, out str);
                if (data.LongLength <= offset)
                {
                    return offset;
                }
                int length = 0;
                for (long i = str; i < offset; ++i)
                {
                    if (0 == data[i])
                    {
                        break;
                    }
                    ++length;
                }
                length = Math.Min(length, BufferSize);
                Array.Copy(data, str, buffer_, 0, length);
                result = Encoding.ASCII.GetString(buffer_, 0, length);
                return offset;
            }

            private long strtoll(ref long offset, byte[] data)
            {
                long length = Math.Min(data.LongLength - offset, BufferSize);
                System.Array.Copy(data, offset, buffer_, 0, length);
                string str = Encoding.ASCII.GetString(buffer_);
                long value;
                if (long.TryParse(str, out value))
                {
                    int i = 0;
                    if (str[0] == '+' || str[0] == '-')
                    {
                        ++i;
                        offset += 1;
                    }
                    for (; i < str.Length; ++i)
                    {
                        if (!char.IsDigit(str[i]))
                        {
                            break;
                        }
                        ++offset;
                    }
                }
                return value;
            }

            public long json_array(long offset, byte[] data, int size, long[] result)
            {
                if (data[offset] != '[')
                {
                    return data.LongLength;
                }
                offset = json_skipws(offset + 1, data);
                for (int i = 0; i < size; ++i)
                {
                    long prev_offset = offset;
                    result[i] = strtoll(ref offset, data);
                    if (prev_offset == offset)
                    {
                        return data.LongLength;
                    }
                    offset = json_skipws(offset, data);
                    if (data[offset] == ']')
                    {
                        return json_skipws(offset + 1, data);
                    }
                    if (data[offset] != ',')
                    {
                        return data.LongLength;
                    }
                    offset = json_skipws(offset + 1, data);
                }
                if (data[offset] != ']')
                {
                    return data.LongLength;
                }
                return json_skipws(offset + 1, data);
            }

            public long json_tensor(long offset, byte[] data, ref Tensor tensor, long bytes_size)
            {
                if (data[offset] != '{')
                {
                    return data.LongLength;
                }
                offset = json_skipws(offset + 1, data);
                int dsize = 0;
                long[] nums = new long[4];
                while (data[offset] != '}')
                {
                    string key;
                    offset = json_string(offset, data, out key);
                    if (data.LongLength <= offset || data[offset] != ':')
                    {
                        return data.LongLength;
                    }
                    offset = json_skipws(offset + 1, data);
                    if (key == "dtype")
                    {
                        string val;
                        offset = json_string(offset, data, out val);
                        if (data.LongLength <= offset)
                        {
                            return offset;
                        }
                        if (!json_dtype(val, out tensor.dtype_, out dsize))
                        {
                            return data.LongLength;
                        }
                    }
                    else if (key == "shape")
                    {
                        Array.Fill(nums, 0);
                        offset = json_array(offset, data, 4, nums);
                        if (data.LongLength <= offset)
                        {
                            return offset;
                        }
                        for (int i = 0; i < 4; ++i)
                        {
                            if (nums[i] < 0 || int.MaxValue < nums[i])
                            {
                                return data.LongLength;
                            }
                        }
                        tensor.shape0_ = (int)nums[0];
                        tensor.shape1_ = (int)nums[1];
                        tensor.shape2_ = (int)nums[2];
                        tensor.shape3_ = (int)nums[3];
                    }
                    else if (key == "data_offsets")
                    {
                        Array.Fill(nums, 0);
                        offset = json_array(offset, data, 2, nums);
                        if (data.LongLength <= offset)
                        {
                            return offset;
                        }
                        if (nums[0] < 0 || nums[1] <= nums[0] || bytes_size < nums[1])
                        {
                            return data.LongLength;
                        }
                        tensor.data_ = nums[0];
                        tensor.size_ = nums[1] - nums[0];
                    }
                    else
                    {
                        return data.LongLength;
                    }
                    if (data[offset] != '}' && data[offset] != ',')
                    {
                        return data.LongLength;
                    }
                    offset = (data[offset] == ',') ? json_skipws(offset + 1, data) : offset;
                }
                if (!validate_shape(dsize, tensor.shape0_, tensor.shape1_, tensor.shape2_, tensor.shape3_, tensor.size_))
                {
                    return data.LongLength;
                }
                return json_skipws(offset + 1, data);
            }

            public long json_metadata(long offset, byte[] data, Tensors tensors)
            {
                if (data[offset] != '{')
                {
                    return data.LongLength;
                }
                offset = json_skipws(offset + 1, data);
                while (data[offset] != '}')
                {
                    Metadata metadata = new Metadata();
                    offset = json_string(offset, data, out metadata.key_);
                    char c = (char)data[offset];
					if (data.LongLength <= offset || data[offset] != ':')
                    {
                        return data.LongLength;
                    }
                    offset = json_skipws(offset + 1, data);
                    offset = json_string(offset, data, out metadata.value_);
                    if (data.LongLength <= offset)
                    {
                        return offset;
                    }
                    tensors.metadata_.Add(metadata);
                    if (data[offset] != '}' && data[offset] != ',')
                    {
                        return data.LongLength;
                    }
                    offset = (data[offset] == ',') ? json_skipws(offset + 1, data) : offset;
                }
                return json_skipws(offset + 1, data);
            }

            public bool json_dtype(string str, out DType dtype, out int dsize)
            {

                switch (str)
                {
                    case "F32":
                        dtype = DType.dt_f32;
                        dsize = 4;
                        break;
                    case "F16":
                        dtype = DType.dt_f16;
                        dsize = 2;
                        break;
                    case "BF16":
                        dtype = DType.dt_bf16;
                        dsize = 2;
                        break;
                    case "F8_E5M2":
                        dtype = DType.dt_f8e5m2;
                        dsize = 1;
                        break;
                    case "F8_E4M3":
                        dtype = DType.dt_f8e4m3;
                        dsize = 1;
                        break;
                    case "I32":
                        dtype = DType.dt_i32;
                        dsize = 2;
                        break;
                    case "I16":
                        dtype = DType.dt_i16;
                        dsize = 2;
                        break;
                    case "I8":
                        dtype = DType.dt_i8;
                        dsize = 1;
                        break;
                    case "U8":
                        dtype = DType.dt_u8;
                        dsize = 1;
                        break;
                    default:
                        dtype = DType.dt_u8;
                        dsize = 0;
                        return false;
                }
                return true;
            }


            public static bool validate_shape(int dsize, int shape0, int shape1, int shape2, int shape3, long length)
            {
                var check = (ref int length, ref int elements, int shape) =>
                {
                    int dim = (shape == 0) ? 1 : shape;
                    if (dim < 0 || elements < dim)
                    {
                        return false;
                    }
                    length *= dim;
                    elements /= dim;
                    return true;
                };

                int expected_length = 1;
                int max_elements = int.MaxValue;
                if (!check(ref expected_length, ref max_elements, shape0))
                {
                    return false;
                }
                if (!check(ref expected_length, ref max_elements, shape1))
                {
                    return false;
                }
                if (!check(ref expected_length, ref max_elements, shape2))
                {
                    return false;
                }
                if (!check(ref expected_length, ref max_elements, shape3))
                {
                    return false;
                }
                return expected_length * dsize == length;
            }

            public static Tensors? parse(byte[] input)
            {
                if (input.LongLength < sizeof(ulong))
                {
                    return null;
                }
                JsonParser parser = new JsonParser();
                ReadOnlySpan<byte> data = input;
                data.Slice(0, sizeof(long));
                long json_size = MemoryMarshal.Cast<byte, long>(data.Slice(0, sizeof(long)))[0];
                if (json_size <= 0 || (input.LongLength - sizeof(long)) < json_size)
                {
                    return null;
                }
                long size = input.LongLength;
                long json = sizeof(long);
                long json_end = json + json_size;
                long bytes = json + json_size;
                long bytes_size = size - sizeof(long) - json_size;
                if (input[json] != '{')
                {
                    return null;
                }
                input[json + json_size - 1] = 0;
                json = parser.json_skipws(json + 1, input);
                Tensors tensors = new Tensors();
                while (json < json_end && input[json] != '}')
                {
                    string key;
                    json = parser.json_string(json, input, out key);
                    if (input.LongLength <= json || input[json] != ':')
                    {
                        return null;
                    }
                    json = parser.json_skipws(json + 1, input);
                    if (key == "__metadata__")
                    {
                        json = parser.json_metadata(json, input, tensors);
                        if (input.LongLength <= json)
                        {
                            return null;
                        }
                    }
                    else
                    {
                        Tensor tensor = new Tensor();
                        json = parser.json_tensor(json, input, ref tensor, bytes_size);
                        if (input.LongLength <= json)
                        {
                            return null;
                        }
                        tensor.name_ = key;
                        tensors.tensors_.Add(tensor);
                    }
                    if (input[json] != '}' && input[json] != ',' && input[json] != '\0')
                    {
                        return null;
                    }
                    json = (input[json] == ',') ? parser.json_skipws(json + 1, input) : json;
                }
                tensors.size_ = input.LongLength;
                tensors.data_ = input;
                return tensors;
            }
        }
    }
}
