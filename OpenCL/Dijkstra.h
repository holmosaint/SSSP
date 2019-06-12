#include <CL/cl.h>
#include "graph.h"

void runDijkstraMultiGPUandCPU( cl_context gpuContext, cl_context cpuContext, GraphData* graph, 
                                int *sourceVertices, long long *outResultCosts, int numResults);