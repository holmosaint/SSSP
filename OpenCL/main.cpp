#include <CL/cl.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <assert.h>
#include <ctime>
#include "graph.h"

using namespace std;

/*
 * This function will compute the shortest-path distance from sourceVertices[n] -> endVertices[n] and store the cost in outResultCosts[n]. The number of results it will compute is given by numResults.z
 */
void runDijkstra(cl_context gpuContext, cl_device_id deviceId, GraphData *graph, int *sourceVertices, long long *outResultCosts, int numResults) {
    
}

void buildGraph(GraphData *graph) {

}

int getSourceVertices(int *sourceVertices) {
    
    return -1;
}

int main(int argc, char **argv) {
    cl_platform_id platform;
    cl_context gpuContext;
    cl_context cpuContext;
    cl_int errNum;

    cl_uint numPlatforms;
    errNum = clGetPlatformIDs(1, &platform, &numPlatforms);
    if(errNum != CL_SUCCESS || numPlatforms <= 0) {
        printf("Failed to find any OpenCL platforms.\n");
        exit(1);
    }

    // create the OpenCL context on GPU devices
    gpuContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
    if(errNum != CL_SUCCESS) {
        printf("No GPU devices found.\n");
        exit(1);            
    }

    // create the OpenCL context on GPU devices
    cpuContext = clCreateContextFromType(0, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
    if(errNum != CL_SUCCESS) {
        printf("No CPU devices found.\n");
        exit(1);
    }

    GraphData graph;
    buildGraph(&graph);

    printf("Vertex Count: %d\n", graph.vertexCount);
    printf("Edge Count: %d\n", graph.edgeCount);

    int sourceNum;
    int *sourceVertices;
    sourceNum = getSourceVertices(sourceVertices);
    assert(sourceNum > 0);

    long long *results = (long long *)malloc(sizeof(long long) * sourceNum * graph.vertexCount);
    clock_t t = clock();
    runDijkstraMultiGPUandCPU(gpuContext, cpuContext, &graph, sourceVertices,
                                                      results, sourceNum);
    printf("Processing time: %.2f\n", (clock() - t) * 1.0 / CLOCKS_PER_SEC);

    free(sourceVertices);
    free(results);
}
