#include <CL/cl.h>

struct GraphData;

void runDijkstraMultiGPUandCPU( cl_context gpuContext, cl_context cpuContext, GraphData* graph, 
                                int *sourceVertices, int *outResultCosts, int numResults);

void runDijkstraMultiGPU( cl_context gpuContext, GraphData* graph, int *sourceVertices,
                          long *outResultCosts, int numResults );
