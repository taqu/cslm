using Silk.NET.OpenCL;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Resources;
using System.Text;
using System.Threading.Tasks;

namespace cslm
{
    public class Device
    {
        const int ARRAY_SIZE = 1000;
        private nint context_ = IntPtr.Zero;
        private nint commandQueue_ = IntPtr.Zero;
        private nint program_ = IntPtr.Zero;
        private nint kernel_ = IntPtr.Zero;
        private nint device_ = IntPtr.Zero;
        private nint[] memObjects_ = new nint[3];

        ~Device()
        {
            Cleanup();
        }

        public unsafe void Initialize()
        {
            CL cl = CL.GetApi();
            context_ = CreateContext(cl);
            if(context_ == IntPtr.Zero)
            {
                return;
            }
            commandQueue_ = CreateCommandQueue(cl, context_, ref device_);
            if (commandQueue_ == IntPtr.Zero)
            {
                Cleanup();
                return;
            }
            Assembly assembly = Assembly.Load("cslm");
            ResourceManager resourceManager = new ResourceManager(assembly.GetName().Name + ".Resources", assembly);
            program_ = CreateProgram(cl, context_, device_, (byte[])resourceManager.GetObject("infer"));
            if (program_ == IntPtr.Zero)
            {
                Cleanup();
                return;
            }

            // Create OpenCL kernel
            kernel_ = cl.CreateKernel(program_, "forward", null);
            if (kernel_ == IntPtr.Zero)
            {
                Cleanup();
                return;
            }

            // Create memory objects that will be used as arguments to
            // kernel.  First create host memory arrays that will be
            // used to store the arguments to the kernel
            float[] result = new float[ARRAY_SIZE];
            float[] a = new float[ARRAY_SIZE];
            float[] b = new float[ARRAY_SIZE];
            for (int i = 0; i < ARRAY_SIZE; i++)
            {
                a[i] = (float)i;
                b[i] = (float)(i * 2);
            }

            if (!CreateMemObjects(cl, context_, memObjects_, a, b))
            {
                Cleanup();
                return;
            }

            // Set the kernel arguments (result, a, b)
            int errNum = cl.SetKernelArg(kernel_, 0, (nuint)sizeof(nint), ref memObjects_[0]);
            errNum |= cl.SetKernelArg(kernel_, 1, (nuint)sizeof(nint), ref memObjects_[1]);
            errNum |= cl.SetKernelArg(kernel_, 2, (nuint)sizeof(nint), ref memObjects_[2]);

            if (errNum != (int)ErrorCodes.Success)
            {
                Console.WriteLine("Error setting kernel arguments.");
                Cleanup();
                return;
            }

            nuint[] globalWorkSize = new nuint[1] { ARRAY_SIZE };
            nuint[] localWorkSize = new nuint[1] { 1 };

            // Queue the kernel up for execution across the array
            errNum = cl.EnqueueNdrangeKernel(commandQueue_, kernel_, 1, (nuint*)null, globalWorkSize, localWorkSize, 0, (nint*)null, (nint*)null);
            if (errNum != (int)ErrorCodes.Success)
            {
                Console.WriteLine("Error queuing kernel for execution.");
                Cleanup();
                return;
            }

            fixed (void* pValue = result)
            {
                // Read the output buffer back to the Host
                errNum = cl.EnqueueReadBuffer(commandQueue_, memObjects_[2], true, 0, ARRAY_SIZE * sizeof(float), pValue, 0, null, null);
                if (errNum != (int)ErrorCodes.Success)
                {
                    Console.WriteLine("Error reading result buffer.");
                    Cleanup();
                    return;
                }
            }

            // Output the result buffer
            for (int i = 0; i < ARRAY_SIZE; i++)
            {
                Console.WriteLine(result[i]);
            }
            Console.WriteLine("Executed program succesfully.");
            Cleanup();
        }


        private void Cleanup()
        {
            CL cl = CL.GetApi();
            for (int i = 0; i < memObjects_.Length; i++)
            {
                if (memObjects_[i] != 0)
                {
                    cl.ReleaseMemObject(memObjects_[i]);
                    memObjects_[i] = 0;
                }
            }
            if (commandQueue_ != 0)
            {
                cl.ReleaseCommandQueue(commandQueue_);
                commandQueue_ = 0;
            }

            if (kernel_ != 0)
            {
                cl.ReleaseKernel(kernel_);
                kernel_ = 0;
            }

            if (program_ != 0)
            {
                cl.ReleaseProgram(program_);
                program_ = 0;
            }

            if (context_ != 0)
            {
                cl.ReleaseContext(context_);
                context_ = 0;
            }
        }

        private static unsafe nint CreateContext(CL cl)
        {
            int errNum = cl.GetPlatformIDs(1, out nint firstPlatformId, out uint numPlatforms);
            if (errNum != (int)ErrorCodes.Success || numPlatforms <= 0)
            {
                Console.WriteLine("Failed to find any OpenCL platforms.");
                return IntPtr.Zero;
            }

            // Next, create an OpenCL context on the platform.  Attempt to
            // create a GPU-based context, and if that fails, try to create
            // a CPU-based context.
            nint[] contextProperties = new nint[]
            {
                (nint)ContextProperties.Platform,
                firstPlatformId,
                0
            };

            fixed (nint* p = contextProperties)
            {
                var context = cl.CreateContextFromType(p, DeviceType.Gpu, null, null, out errNum);
                if (errNum != (int)ErrorCodes.Success)
                {
                        return IntPtr.Zero;
                }

                return context;
            }
        }

        private static unsafe nint CreateCommandQueue(CL cL, nint context, ref nint device)
        {
            int errNum = cL.GetContextInfo(context, ContextInfo.Devices, 0, null, out nuint deviceBufferSize);
            if (errNum != (int)ErrorCodes.Success)
            {
                return IntPtr.Zero;
            }

            if (deviceBufferSize <= 0)
            {
                return IntPtr.Zero;
            }

            nint[] devices = new nint[deviceBufferSize / (nuint)sizeof(nuint)];
            fixed (void* pValue = devices)
            {
                int er = cL.GetContextInfo(context, ContextInfo.Devices, deviceBufferSize, pValue, null);

            }
            if (errNum != (int)ErrorCodes.Success)
            {
                devices = null;
                return IntPtr.Zero;
            }

            // In this example, we just choose the first available device.  In a
            // real program, you would likely use all available devices or choose
            // the highest performance device based on OpenCL device queries
            var commandQueue = cL.CreateCommandQueue(context, devices[0], CommandQueueProperties.None, null);
            if (commandQueue == IntPtr.Zero)
            {
                return IntPtr.Zero;
            }

            device = devices[0];
            return commandQueue;
        }

        public static unsafe nint CreateProgram(CL cl, nint context, nint device, byte[]? bytes)
        {
            if(null == bytes)
            {
                return IntPtr.Zero;
            }
            string clStr = Encoding.UTF8.GetString(bytes);

            nint program = cl.CreateProgramWithSource(context, 1, new string[] { clStr }, null, null);
            if (program == IntPtr.Zero)
            {
                return IntPtr.Zero;
            }

            int errNum = cl.BuildProgram(program, 0, null, (byte*)null, null, null);

            if (errNum != (int)ErrorCodes.Success)
            {
                _ = cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.BuildLog, 0, null, out nuint buildLogSize);
                byte[] log = new byte[buildLogSize / (nuint)sizeof(byte)];
                fixed (void* pValue = log)
                {
                    cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.BuildLog, buildLogSize, pValue, null);
                }
                string? build_log = System.Text.Encoding.UTF8.GetString(log);

                //Console.WriteLine("Error in kernel: ");
                Console.WriteLine("=============== OpenCL Program Build Info ================");
                Console.WriteLine(build_log);
                Console.WriteLine("==========================================================");

                cl.ReleaseProgram(program);
                return IntPtr.Zero;
            }
            return program;
        }

        static unsafe bool CreateMemObjects(CL cl, nint context, nint[] memObjects, float[] a, float[] b)
        {
            fixed (void* pa = a)
            {
                memObjects[0] = cl.CreateBuffer(context, MemFlags.ReadOnly | MemFlags.CopyHostPtr, sizeof(float) * ARRAY_SIZE, pa, null);
            }

            fixed (void* pb = b)
            {
                memObjects[1] = cl.CreateBuffer(context, MemFlags.ReadOnly | MemFlags.CopyHostPtr, sizeof(float) * ARRAY_SIZE, pb, null);
            }

            memObjects[2] = cl.CreateBuffer(context, MemFlags.ReadWrite, sizeof(float) * ARRAY_SIZE, null, null);

            if (memObjects[0] == IntPtr.Zero || memObjects[1] == IntPtr.Zero || memObjects[2] == IntPtr.Zero)
            {
                Console.WriteLine("Error creating memory objects.");
                return false;
            }

            return true;
        }
    }
}
