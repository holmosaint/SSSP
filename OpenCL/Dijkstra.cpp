#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include "device.h"
#include "Dijkstra.h"
#include <CL/cl.h>

using namespace std;

#define checkError(a, b) checkErrorFileLine(a, b, __FILE__ , __LINE__)

/*
 * Macro Options
 * Number of async loop iterations before attempting to read results back
 */
#define NUM_ASYNCHRONOUS_ITERATIONS 10

/*
 * function declaration
 */
int roundWorkSizeUp(int groupSize, int globalSize);

/*
 * Check for error condition and exit if found.  Print file and line number
 * of error. (from NVIDIA SDK)
 */ 
void checkErrorFileLine(int errNum, int expected, const char* file, const int lineNumber)
{
    if (errNum != expected)
    {
        cerr << "Line " << lineNumber << " in File " << file << endl;
        exit(1);
    }
}

/*
 * Check whether the mask array is empty.  This tells the algorithm whether
 * it needs to continue running or not.
 */
bool maskArrayEmpty(int *maskArray, int count)
{
    for(int i = 0; i < count; i++ )
    {
        if (maskArray[i] == 1)
        {
            return false;
        }
    }

    return true;
}


/*
 * global variable
 */
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

cl_device_id getDev(cl_context cxGPUContext, unsigned int nr) {
    size_t szParmDataBytes;
    cl_device_id *cdDevices;

    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);

    if(szParmDataBytes / sizeof(cl_device_id) < nr)
        return (cl_device_id) - 1;

    cdDevices = (cl_device_id *) malloc(szParmDataBytes);

    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);

    cl_device_id device = cdDevices[nr];
    free(cdDevices);

    return device;
}

/*
 * Gets the id of the first device from the context (from the NVIDIA SDK)
 */
cl_device_id getFirstDev(cl_context cxGPUContext)
{
    size_t szParmDataBytes;
    cl_device_id* cdDevices;

    // get the list of GPU devices associated with context
    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
    cdDevices = (cl_device_id*) malloc(szParmDataBytes);

    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);

    cl_device_id first = cdDevices[0];
    free(cdDevices);

    return first;
}

/*
 * Allocate memory for input CUDA buffers and copy the data into device memory
 */
void allocateOCLBuffers(cl_context gpuContext, cl_command_queue commandQueue, GraphData *graph,
                        cl_mem *vertexArrayDevice, cl_mem *edgeArrayDevice, cl_mem *weightArrayDevice,
                        cl_mem *maskArrayDevice, cl_mem *costArrayDevice, cl_mem *updatingCostArrayDevice,
                        size_t globalWorkSize)
{
    cl_int errNum;
    cl_mem hostVertexArrayBuffer;
    cl_mem hostEdgeArrayBuffer;
    cl_mem hostWeightArrayBuffer;

    // First, need to create OpenCL Host buffers that can be copied to device buffers
    hostVertexArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                           sizeof(int) * graph->vertexCount, graph->vertexArray, &errNum);
    checkError(errNum, CL_SUCCESS);

    hostEdgeArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                           sizeof(int) * graph->edgeCount, graph->edgeArray, &errNum);
    checkError(errNum, CL_SUCCESS);

    hostWeightArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                           sizeof(int) * graph->edgeCount, graph->weightArray, &errNum);
    checkError(errNum, CL_SUCCESS);

    // Now create all of the GPU buffers
    *vertexArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(int) * globalWorkSize, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *edgeArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(int) * graph->edgeCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *weightArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(int) * graph->edgeCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *maskArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(int) * globalWorkSize, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *costArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(int) * globalWorkSize, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    *updatingCostArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(int) * globalWorkSize, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);

    // Now queue up the data to be copied to the device
    errNum = clEnqueueCopyBuffer(commandQueue, hostVertexArrayBuffer, *vertexArrayDevice, 0, 0,
                                 sizeof(int) * graph->vertexCount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);

    errNum = clEnqueueCopyBuffer(commandQueue, hostEdgeArrayBuffer, *edgeArrayDevice, 0, 0,
                                 sizeof(int) * graph->edgeCount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);

    errNum = clEnqueueCopyBuffer(commandQueue, hostWeightArrayBuffer, *weightArrayDevice, 0, 0,
                                 sizeof(int) * graph->edgeCount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);

    clReleaseMemObject(hostVertexArrayBuffer);
    clReleaseMemObject(hostEdgeArrayBuffer);
    clReleaseMemObject(hostWeightArrayBuffer);
}

/*
 * Initialize OpenCL buffers for single run of Dijkstra
 */
void initializeOCLBuffers(cl_command_queue commandQueue, cl_kernel initializeKernel, GraphData *graph,
                          size_t maxWorkGroupSize)
{
    cl_int errNum;
    // Set # of work items in work group and total in 1 dimensional range
    size_t localWorkSize = maxWorkGroupSize;
    size_t globalWorkSize = roundWorkSizeUp(localWorkSize, graph->vertexCount);

    errNum = clEnqueueNDRangeKernel(commandQueue, initializeKernel, 1, NULL, &globalWorkSize, &localWorkSize,
                                    0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);
}


/*
 * Round the local work size up to the next multiple of the size
 */
int roundWorkSizeUp(int groupSize, int globalSize) {
    int remainder = globalSize % groupSize;
    if(remainder == 0)
        return globalSize;
    return globalSize + groupSize - remainder;
}


cl_program loadAndBuildProgram(cl_context gpuContext, const char *fileName) {
    pthread_mutex_lock(&mutex);

    cl_int errNum;
    cl_program program;

    // Load the OpenCL source code from the .cl file
    std::ifstream kernelFile(fileName, std::ios::in);
    if(!kernelFile.is_open()) {
        printf("Error: Failed to open file for reading %s\n", fileName);
        exit(1);
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *source = srcStdStr.c_str();
    if(source == NULL) {
        printf("Error: Can not read source file content.\n");
        exit(1);
    }
    // printf("%s\n", source);

    // Create the program for all GPUs in the context
    program = clCreateProgramWithSource(gpuContext, 1, (const char **)&source, NULL, &errNum);
    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(errNum != CL_SUCCESS) {
        char cBuildLog[10240];
        clGetProgramBuildInfo(program, getFirstDev(gpuContext), CL_PROGRAM_BUILD_LOG, sizeof(cBuildLog), cBuildLog, NULL);
        cerr << cBuildLog << endl;
        printf("Error: can not build program from source.\n");
        exit(1);
    }

    pthread_mutex_unlock(&mutex);
    return program;
}


/*
 * Run Dijkstra's shortest path on the GraphData provided to this function.  This
 * function will compute the shortest path distance from sourceVertices[n] ->
 * endVertices[n] and store the cost in outResultCosts[n].  The number of results
 * it will compute is given by numResults.
 * This function will run the algorithm on as many GPUs as is available along with
 * the CPU.  It will create N threads, one for each device, and chunk the workload up to perform
 * (numResults / N) searches per device.
 * */

void runDijkstra(cl_context context, cl_device_id deviceId, GraphData *graph, int *sourceVertices, int *outResultCosts, int numResults) {
    cl_int errNum;
    cl_command_queue commandQueue;
    commandQueue = clCreateCommandQueue(context, deviceId, 0, &errNum);
    printf("Create command queue for device %d at 0x%x\n", deviceId, &commandQueue);
    if(errNum != CL_SUCCESS) {
        printf("ErrorL: Can not build command queue!\n");
        exit(1);
    }

    cl_program program = loadAndBuildProgram(context, "dijkstra.cl");
    if(program <= 0) {
        printf("Error: Can not load program!\n");
        exit(1);
    }

    // get the max workgroup size
    size_t maxWorkGroupSize;
    clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
    if(errNum != CL_SUCCESS) {
        printf("Error: Cannot get max work group size!\n");
        exit(1);
    }
    printf("Max work group size: %d\n", maxWorkGroupSize);
    printf("Computing %d results.\n", numResults);

    // set # of work items in work group and total in 1 dimension
    size_t localWorkSize = maxWorkGroupSize;
    size_t globalWorkSize = roundWorkSizeUp(localWorkSize, graph->vertexCount);

    cl_mem vertexArrayDevice;
    cl_mem edgeArrayDevice;
    cl_mem weightArrayDevice;
    cl_mem maskArrayDevice;
    cl_mem costArrayDevice;
    cl_mem updatingCostArrayDevice;

    // allocate buyffers in device memory
    allocateOCLBuffers(context, commandQueue, graph, &vertexArrayDevice, &edgeArrayDevice, &weightArrayDevice, 
                    &maskArrayDevice, &costArrayDevice, &updatingCostArrayDevice ,globalWorkSize);

    // create the kernels
    cl_kernel initializeBuffersKernel;
    initializeBuffersKernel = clCreateKernel(program, "initializeBuffers", &errNum);
    if(errNum != CL_SUCCESS) {
        printf("Error: can not create initialize buffer kernel: %d\n", errNum);
        exit(1);
    }
    printf("Create Kernel at device %d\n", deviceId);

    // set the args
    errNum |= clSetKernelArg(initializeBuffersKernel, 0, sizeof(cl_mem), &maskArrayDevice);
    errNum |= clSetKernelArg(initializeBuffersKernel, 1, sizeof(cl_mem), &costArrayDevice);
    errNum |= clSetKernelArg(initializeBuffersKernel, 2, sizeof(cl_mem), &updatingCostArrayDevice);
    errNum |= clSetKernelArg(initializeBuffersKernel, 4, sizeof(int), &graph->vertexCount);
    if(errNum != CL_SUCCESS) {
        printf("Error: can not set args for initialize Buffers Kernel.\n");
        exit(1);
    }

    // kernel 1
    cl_kernel ssspKernel1;
    ssspKernel1 = clCreateKernel(program, "DijkstraKernel1", &errNum);
    if(errNum != CL_SUCCESS) {
        printf("Error: can not create kernel 1.\n");
        exit(1);
    }

    // set the args
    errNum |= clSetKernelArg(ssspKernel1, 0, sizeof(cl_mem), &vertexArrayDevice);
    errNum |= clSetKernelArg(ssspKernel1, 1, sizeof(cl_mem), &edgeArrayDevice);
    errNum |= clSetKernelArg(ssspKernel1, 2, sizeof(cl_mem), &weightArrayDevice);
    errNum |= clSetKernelArg(ssspKernel1, 3, sizeof(cl_mem), &maskArrayDevice);
    errNum |= clSetKernelArg(ssspKernel1, 4, sizeof(cl_mem), &costArrayDevice);
    errNum |= clSetKernelArg(ssspKernel1, 5, sizeof(cl_mem), &updatingCostArrayDevice);
    errNum |= clSetKernelArg(ssspKernel1, 6, sizeof(int), &graph->vertexCount);
    errNum |= clSetKernelArg(ssspKernel1, 7, sizeof(int), &graph->edgeCount);
    if(errNum != CL_SUCCESS) {
        printf("Error: can not set args for kernel 1.\n");
        exit(1);
    }

    cl_kernel ssspKernel2;
    ssspKernel2 = clCreateKernel(program, "DijkstraKernel2", &errNum);
    if(errNum != CL_SUCCESS) {
        printf("Error: can not create kernel 2.\n");
        exit(1);
    }
    errNum |= clSetKernelArg(ssspKernel2, 0, sizeof(cl_mem), &vertexArrayDevice);
    errNum |= clSetKernelArg(ssspKernel2, 1, sizeof(cl_mem), &edgeArrayDevice);
    errNum |= clSetKernelArg(ssspKernel2, 2, sizeof(cl_mem), &weightArrayDevice);
    errNum |= clSetKernelArg(ssspKernel2, 3, sizeof(cl_mem), &maskArrayDevice);
    errNum |= clSetKernelArg(ssspKernel2, 4, sizeof(cl_mem), &costArrayDevice);
    errNum |= clSetKernelArg(ssspKernel2, 5, sizeof(cl_mem), &updatingCostArrayDevice);
    errNum |= clSetKernelArg(ssspKernel2, 6, sizeof(int), &graph->vertexCount);
    if(errNum != CL_SUCCESS) {
        printf("Error: can not set args for kernel 2.\n");
        exit(1);
    }

    int *maskArrayHost = (int *) malloc(sizeof(int) * graph->vertexCount);

    for(int i = 0; i < numResults; ++i) {
        errNum |= clSetKernelArg(initializeBuffersKernel, 3, sizeof(int), &sourceVertices[i]);
        if(errNum != CL_SUCCESS) {
            printf("Error: can not set the 3rd arg for initialize Buffers Kernel.\n");
            exit(1);
        }

        // Initialize mask array to false, C and U to infiniti
        initializeOCLBuffers(commandQueue, initializeBuffersKernel, graph, maxWorkGroupSize);

        // Read mask array from device -> host
        cl_event readDone;
        errNum = clEnqueueReadBuffer(commandQueue, maskArrayDevice, CL_FALSE, 0, sizeof(int) * graph->vertexCount, 
                        maskArrayHost, 0, NULL, &readDone);
        if(errNum != CL_SUCCESS) {
            printf("Error: Can not read from mask array.\n");
            exit(1);
        }

        clWaitForEvents(1, &readDone);

        while(!maskArrayEmpty(maskArrayHost, graph->vertexCount)) {
            // In order to improve performance, we run some number of iterations
            // without reading the results.  This might result in running more iterations
            // than necessary at times, but it will in most cases be faster because
            // we are doing less stalling of the GPU waiting for results.
            for(int asynIter = 0; asynIter < NUM_ASYNCHRONOUS_ITERATIONS; ++asynIter) {
                size_t localWorkSize = maxWorkGroupSize;
                size_t globalWorkSize = roundWorkSizeUp(localWorkSize, graph->vertexCount);

                // execute the kernel
                errNum = clEnqueueNDRangeKernel(commandQueue, ssspKernel1, 1, 0, &globalWorkSize, &localWorkSize, 
                                0, NULL, NULL);
                    
                if(errNum != CL_SUCCESS) {
                    printf("Error: in execute kernel 1.\n");
                    exit(1);
                }

                errNum = clEnqueueNDRangeKernel(commandQueue, ssspKernel2, 1, 0, &globalWorkSize, &localWorkSize, 
                                0, NULL, NULL);

                if(errNum != CL_SUCCESS) {
                    printf("Error: in execute kernel 1.\n");
                    exit(1);
                }
                
                errNum = clEnqueueReadBuffer(commandQueue, maskArrayDevice, CL_FALSE, 0, sizeof(int) * graph->vertexCount, 
                                maskArrayHost, 0, NULL, &readDone) ;
                if(errNum != CL_SUCCESS) {
                    printf("Error: Can not read from mask array.\n");
                    exit(1);                                                      
                }
                clWaitForEvents(1, &readDone);
            }
        }

        // copy the results back
        errNum = clEnqueueReadBuffer(commandQueue, costArrayDevice, CL_FALSE, 0, sizeof(int) * graph->vertexCount, 
                        &outResultCosts[i * graph->vertexCount], 0, NULL, &readDone);
                    
        if(errNum != CL_SUCCESS) {
            printf("Error: can not read from cost array.\n");
            exit(1);
        }
        clWaitForEvents(1, &readDone);
    }

    free(maskArrayHost);

    clReleaseMemObject(vertexArrayDevice);
    clReleaseMemObject(edgeArrayDevice);
    clReleaseMemObject(weightArrayDevice);
    clReleaseMemObject(maskArrayDevice);
    clReleaseMemObject(costArrayDevice);
    clReleaseMemObject(updatingCostArrayDevice);

    clReleaseKernel(initializeBuffersKernel);
    clReleaseKernel(ssspKernel1);
    clReleaseKernel(ssspKernel2);

    clReleaseCommandQueue(commandQueue);
    clReleaseProgram(program);
    printf("Computed %d results.\n", numResults);
}

void dijkstraThread(DevicePlan *plan) {
    runDijkstra(plan->context, plan->deviceId, plan->graph, plan->sourceVertices, plan->outResultCosts, plan->numResults);
}

void runDijkstraMultiGPU( cl_context gpuContext, GraphData* graph, int *sourceVertices,
                          int *outResultCosts, int numResults )
{

    // Find out how many GPU's to compute on all available GPUs
    cl_int errNum;
    size_t deviceBytes;
    cl_uint deviceCount;

    errNum = clGetContextInfo(gpuContext, CL_CONTEXT_DEVICES, 0, NULL, &deviceBytes);
    checkError(errNum, CL_SUCCESS);
    deviceCount = (cl_uint)deviceBytes/sizeof(cl_device_id);

    if (deviceCount == 0)
    {
        cerr << "ERROR: no GPUs present!" << endl;
        return;
    }

    DevicePlan *devicePlans = (DevicePlan*) malloc(sizeof(DevicePlan) * deviceCount);
    pthread_t *threadIDs = (pthread_t*) malloc(sizeof(pthread_t) * deviceCount);

    // Divide the workload out per device
    int resultsPerDevice = numResults / deviceCount;

    int offset = 0;

    for (unsigned int i = 0; i < deviceCount; i++)
    {
        devicePlans[i].context = gpuContext;
        devicePlans[i].deviceId = getDev(gpuContext, i);;
        devicePlans[i].graph = graph;
        devicePlans[i].sourceVertices = &sourceVertices[offset];
        devicePlans[i].outResultCosts = &outResultCosts[offset * graph->vertexCount];
        devicePlans[i].numResults = resultsPerDevice;

        offset += resultsPerDevice;
    }

    // Add any remaining work to the last GPU
    if (offset < numResults)
    {
        devicePlans[deviceCount - 1].numResults += (numResults - offset);
    }

    // Launch all the threads
    for (unsigned int i = 0; i < deviceCount; i++)
    {
        pthread_create(&threadIDs[i], NULL, (void* (*)(void*))dijkstraThread, (void*)(devicePlans + i));
    }

    // Wait for the results from all threads
    for (unsigned int i = 0; i < deviceCount; i++)
    {
        pthread_join(threadIDs[i], NULL);
    }

    free (devicePlans);
    free (threadIDs);
}

void runDijkstraMultiGPUandCPU( cl_context gpuContext, cl_context cpuContext, GraphData* graph, 
                                int *sourceVertices, int *outResultCosts, int numResults) {
    float ratioCPUtoGPU = 2.26;     // CPU seems to run it at 2.26X on GT120 GPU
    
    // Find out how many GPU's to compute on all available GPUs
    cl_int errNum;
    size_t deviceBytes;
    cl_uint gpuDeviceCount;
    cl_uint cpuDeviceCount;

    errNum = clGetContextInfo(gpuContext, CL_CONTEXT_DEVICES, 0, NULL, &deviceBytes);
    if(errNum != CL_SUCCESS) {
        printf("Error: Cannot get GPU information.\n");
        exit(1);
    }
    gpuDeviceCount = (cl_uint)deviceBytes / sizeof(cl_device_id);

    if(gpuDeviceCount == 0) {
        printf("Error: No GPU found!\n");
        exit(1);
    }
    printf("%d GPUs found.\n", gpuDeviceCount);

    errNum = clGetContextInfo(cpuContext, CL_CONTEXT_DEVICES, 0, NULL, &deviceBytes);
    if(errNum != CL_SUCCESS) {
        printf("Error: Cannot get CPU information.\n");
        exit(1);
    }
    cpuDeviceCount = (cl_uint)deviceBytes / sizeof(cl_device_id);

    if(cpuDeviceCount == 0) {
        printf("Error: No CPU found.\n");
        exit(1);
    }
    printf("%d CPUs found.\n", cpuDeviceCount);
    
    cl_uint totalDeviceCount = gpuDeviceCount + cpuDeviceCount;
    
    DevicePlan *devicePlans = (DevicePlan *)    malloc(sizeof(DevicePlan) * totalDeviceCount);
    pthread_t *threadIDs = (pthread_t *)malloc(sizeof(pthread_t) * totalDeviceCount);

    int gpuResults, cpuResults;
    gpuResults = numResults / ratioCPUtoGPU;
    cpuResults = numResults - gpuResults;
    printf("Number of results on GPU: %d\nNumber of results on CPU: %d\n", 
            gpuResults, cpuResults);

    int resultsPerGPU = gpuResults / gpuDeviceCount; // seems wrong?
    
    int offset = 0;
    int curDevice = 0;
    for(int i = 0;i < gpuDeviceCount; ++i) {
        devicePlans[curDevice].context = gpuContext;
        devicePlans[curDevice].deviceId = getDev(gpuContext, i);
        devicePlans[curDevice].graph = graph;
        devicePlans[curDevice].sourceVertices = &sourceVertices[offset];
        devicePlans[curDevice].outResultCosts = &outResultCosts[offset * graph->vertexCount];
        devicePlans[curDevice].numResults = resultsPerGPU;

        offset += resultsPerGPU;
        ++curDevice;
    }

    int resultsPerCPU = cpuResults;
    for(int i = 0;i < cpuDeviceCount; ++i) {
        devicePlans[curDevice].context = cpuContext;
        devicePlans[curDevice].deviceId = getDev(cpuContext, i);
        devicePlans[curDevice].graph = graph;
        devicePlans[curDevice].sourceVertices = &sourceVertices[offset];
        devicePlans[curDevice].outResultCosts = &outResultCosts[offset * graph->vertexCount];
        devicePlans[curDevice].numResults = resultsPerCPU;

        offset += resultsPerCPU;
        ++curDevice;
    }

    // Add any remaining work to the last CPU
    if(offset < numResults)
        devicePlans[totalDeviceCount - 1].numResults += (numResults - offset);

    for(int i = 0;i < totalDeviceCount; ++i)
        pthread_create(&threadIDs[i], NULL, (void *(*)(void *))dijkstraThread, (void *)(devicePlans + i));

    for(int i = 0;i < totalDeviceCount; ++i)
        pthread_join(threadIDs[i], NULL);

    free(devicePlans);
    free(threadIDs);
}
