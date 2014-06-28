/* Paralel Kernel Independent Fast Multipole Method
Copyright (C) 2010 George Biros, Harper Langston, Ilya Lashuk
Copyright (C) 2010, Aparna Chandramowlishwaran, Aashay Shingrapure, Rich Vuduc

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2, or (at your option)
any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License
along with this program; see the file COPYING.  If not, write to the Free
Software Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
02111-1307, USA.  */

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(-1);
    }
}



int devcheck(int gpudevice)
{
    int device_count=0;
    int device;  // used with cudaGetDevice() to verify cudaSetDevice()

    // get the number of non-emulation devices detected
    cudaGetDeviceCount( &device_count);
    if (gpudevice >= device_count)
    {
        printf("gpudevice = %d , valid devices = [ ", gpudevice);
   for (int i=0; i<device_count; i++)
      printf("%d ", i);
   printf("] ... exiting \n");
        exit(1);
    }
    cudaError_t cudareturn;
    cudaDeviceProp deviceProp;

    // cudaGetDeviceProperties() is also demonstrated in the deviceQuery/ 
    // of the sdk projects directory
    cudaGetDeviceProperties(&deviceProp, gpudevice);
    printf("[deviceProp.major.deviceProp.minor] = [%d.%d]\n",
     deviceProp.major, deviceProp.minor);

    if (deviceProp.major > 999)
    {
        printf("warning, CUDA Device Emulation (CPU) detected, exiting\n");
                exit(1);
    }

    // choose a cuda device for kernel execution
    cudareturn=cudaSetDevice(gpudevice);
    if (cudareturn == cudaErrorInvalidDevice)
   {
        perror("cudaSetDevice returned cudaErrorInvalidDevice");
    }
    else
    {
        // double check that device was properly selected
        cudaGetDevice(&device);
        printf("cudaGetDevice()=%d\n",device);
    }
    return(0);
}

