#include <CL/cl.h>
#include "graph.h"

typedef struct
{
    // Context
    cl_context context;
    
    // Device number to run algorithm on
    cl_device_id deviceId;
    
    // Pointer to graph data
    GraphData *graph;
            
    // Source vertex indices to process
    int *sourceVertices;
            
    // End vertex indices to process
    int *endVertices;
            
    // Results of processing
    int *outResultCosts;
            
    // Number of results
    int numResults;                                         
} DevicePlan;
