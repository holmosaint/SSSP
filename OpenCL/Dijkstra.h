#include <CL/cl.h>

struct GraphData;

void runDijkstraMultiGPUandCPU( cl_context gpuContext, cl_context cpuContext, GraphData* graph, 
                                int *sourceVertices, long long *outResultCosts, int numResults);