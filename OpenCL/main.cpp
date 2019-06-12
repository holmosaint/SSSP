#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <vector>
#include <deque>
#include <assert.h>
#include <ctime>
#include "graph.h"
#include "Dijkstra.h"

// using namespace std;
struct node_struct {
	int id;
	int dis;
	node_struct(int _id, int _dis) :id(_id), dis(_dis) {}
};

void buildGraph(GraphData *graph, char *graphFile) {
    std::ifstream infile;
	infile.open(graphFile, ios::in);
	if (!infile.is_open()) {
		cout << "Can not open the gr file!" << endl;
		exit(1);
	}
	std::string line;
	int v_cnt, arc_cnt;
	std::deque<node_struct> *node_matrix = NULL;
	while (getline(infile, line)) {
		if (line[0] == 'c')
			continue;
		stringstream lineStream(line);
		string type, tmp;
		int src, target, w;
		lineStream >> type;
		if (type == "p") {
			lineStream >> tmp;
			lineStream >> v_cnt >> arc_cnt;
			node_matrix = new deque<node_struct>[v_cnt];
		}
		else {
			lineStream >> src >> target >> w;
			// printf("From src %d to target %d, weights %d\n", src, target, w);
			node_matrix[src - 1].push_back(node_struct(target - 1, w));
		}
	}
	infile.close();

    // convert to the target data struct
    graph->vertexCount = v_cnt;
    graph->edgeCount = arc_cnt;
    graph->vertexArray = (int *)malloc(sizeof(int) * v_cnt);
    graph->edgeArray = (int *)malloc(sizeof(int) * arc_cnt);
    graph->weightArray = (int *)malloc(sizeof(int) * arc_cnt);

    int offset = 0;
    for(int i = 0;i < v_cnt; ++i) {
        // vertex array
        graph->vertexArray[i] = offset;
        int arc_num = node_matrix[i].size();
        for(int j = 0;j < arc_num; ++j) {
            graph->edgeArray[offset + j] = node_matrix[i][j].id;
            graph->weightArray[offset + j] = node_matrix[i][j].dis;
        }
        offset += arc_num;
    }

    free(node_matrix);
}

int getSourceVertices(int *sourceVertices, char *sourceFile) {
    int sourceCount = 0;
    std::vector<int> src_node;
    std::ifstream infile;
	infile.open(sourceFile, ios::in);
	if (!infile.is_open()) {
		cout << "Can not open the ss file!" << endl;
		exit(1);
	}
    std::string line;
	while (getline(infile, line)) {
		if (line[0] == 'c')
			continue;
		std::stringstream lineStream(line);
		std::string type, tmp;
		int src;
		lineStream >> type;
		if (type == "p") {
			lineStream >> tmp >> tmp >> tmp;
			lineStream >> sourceCount;
		}
		else if (type == "s"){
			lineStream >> src;
			src_node.push_back(src - 1);
		}
	}
	assert(sourceCount == src_node.size());
    infile.close();

    sourceVertices = (int *)malloc(sizeof(int) * sourceCount);
    std::copy(src_node.begin(), src_node.end(), sourceVertices);
    return sourceCount;
}

void releaseGraph(GraphData *graph) {
    free(graph->vertexArray);
    free(graph->edgeArray);
    free(graph->weightArray);
}

int main(int argc, char **argv) {
    cl_platform_id platform;
    cl_context gpuContext;
    cl_context cpuContext;
    cl_int errNum;
    clock_t t

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

    t = clock();
    GraphData graph;
    buildGraph(&graph);
    printf("Building graph time: %.2f\n", (clock() - t) * 1.0 / CLOCKS_PER_SEC);

    printf("Vertex Count: %d\n", graph.vertexCount);
    printf("Edge Count: %d\n", graph.edgeCount);

    int sourceNum;
    int *sourceVertices;
    sourceNum = getSourceVertices(sourceVertices);
    assert(sourceNum > 0);

    long long *results = (long long *)malloc(sizeof(long long) * sourceNum * graph.vertexCount);
    t = clock();
    runDijkstraMultiGPUandCPU(gpuContext, cpuContext, &graph, sourceVertices,
                                                      results, sourceNum);
    printf("Processing time: %.2f\n", (clock() - t) * 1.0 / CLOCKS_PER_SEC);

    releaseGraph(graph);
    free(sourceVertices);
    free(results);
}
