// to initialize the buffer
__kernel void initializeBuffers(__global int *maskArray, __global long *costArray, __global long *updatingCostArray, __global int *vertexArray,
            __global int *edgeArray,
            __global int *weightArray,
			int sourceVertex, int vertexCount, int edgeCount) {
    int tid = get_global_id(0);

    if(sourceVertex == tid) {
        maskArray[tid] = 0;
        costArray[tid] = 0;
        updatingCostArray[tid] = 0;
    }
    else {
        maskArray[tid] = 0;
		int edgeStart = vertexArray[tid];
        int edgeEnd;
        if(tid + 1 < vertexCount)
            edgeEnd = vertexArray[tid + 1];
        else edgeEnd = edgeCount;
		
		// binary search
		int mid = (edgeStart + edgeEnd) / 2;
		int beg = edgeStart, end = edgeEnd;
		while(beg < end) {
			mid = (beg + end) / 2;
			int nid = edgeArray[mid];
			if(nid == sourceVertex) {
				maskArray[tid] = 1;
				break;
			}
			if(nid < sourceVertex) {
				beg = mid + 1;
			}
			else end = mid;
		}
        costArray[tid] = LONG_MAX;
        updatingCostArray[tid] = LONG_MAX;
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

		bool update = false;
        for(int edge = edgeStart; edge < edgeEnd; ++edge) {
            int nid = edgeArray[edge];
			if(costArray[nid] == LONG_MAX)
				continue;
            long update = updatingCostArray[tid];
            long new_cost = costArray[nid] + weightArray[edge];
            if(update > new_cost) {
				updatingCostArray[tid] = new_cost;
			}
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
                              int vertexCount, int edgeCount) {
    int tid = get_global_id(0);

    if(costArray[tid] > updatingCostArray[tid]) {
        costArray[tid] = updatingCostArray[tid];
		
		int edgeStart = vertexArray[tid];
        int edgeEnd;
        if(tid + 1 < vertexCount)
            edgeEnd = vertexArray[tid + 1];
        else edgeEnd = edgeCount;

		for(int edge = edgeStart; edge < edgeEnd; ++edge) {
	        int nid = edgeArray[edge];
            maskArray[nid] = 1;
        }
    }
    else updatingCostArray[tid] = costArray[tid];
}
