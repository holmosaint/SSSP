// to initialize the buffer
__kernel void initializeBuffers(__global int *maskArray, __global long *costArray, __global long *updatingCostArray, int sourceVertex, int vertexCount) {
    int tid = get_global_id(0);

    if(sourceVertex == tid) {
        maskArray[tid] = 1;
        costArray[tid] = 0;
        updatingCostArray[tid] = 0;
    }
    else {
        maskArray[tid] = 0;
        costArray[tid] = INT_MAX;
        updatingCostArray[tid] = INT_MAX;
    }
}

// first phase
__kernel void DijkstraKernel1(__global int *vertexArray,
                              __global int *edgeArray,
                              __global int *weightArray,
                              __global int *maskArray,
                              __global long *costArray, 
                              __global long *updatingCostArray,
                              int vertexCount, 
                              int edgeCount) {
    int tid = get_global_id(0);
    
    if(maskArray[tid] != 0) {
        maskArray[tid] = 0;
        
        int edgeStart = vertexArray[tid];
        int edgeEnd;
        if(tid + 1 < vertexCount)
            edgeEnd = vertexArray[tid + 1];
        else edgeEnd = edgeCount;

        for(int edge = edgeStart; edge < edgeEnd; ++edge) {
            int nid = edgeArray[edge];
            long update = updatingCostArray[nid];
            long new_cost = costArray[tid] + weightArray[edge];
            if(update > new_cost)
                updatingCostArray[nid] = new_cost;
        }
    }
}

// second phase
__kernel void DijkstraKernel2(__global int *vertexArray, 
                              __global int *edgeArray, 
                              __global int *weightArray, 
                              __global int *maskArray, 
                              __global long *costArray, 
                              __global long *updatingCostArray, 
                              int vertexCount) {
    int tid = get_global_id(0);

    if(costArray[tid] > updatingCostArray[tid]) {
        costArray[tid] = updatingCostArray[tid];
        maskArray[tid] = 1;
    }

    else updatingCostArray[tid] = costArray[tid];
}
