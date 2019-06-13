#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <vector>
#include <deque>
#include <assert.h>
#include <ctime>
#include <cstring>
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
    printf("Graph file: %s\n", graphFile);
	infile.open(graphFile, std::ios::in);
	if (!infile.is_open()) {
        printf("Can not open the gr file!\n");
        exit(1);
	}
	std::string line;
	int v_cnt, arc_cnt;
	std::deque<node_struct> *node_matrix = NULL;
	while (getline(infile, line)) {
		if (line[0] == 'c')
			continue;
        std::stringstream lineStream(line);
        std::string type, tmp;
		int src, target, w;
		lineStream >> type;
		if (type == "p") {
			lineStream >> tmp;
			lineStream >> v_cnt >> arc_cnt;
			node_matrix = new std::deque<node_struct>[v_cnt];
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
	int max_edge = 0;
    for(int i = 0;i < v_cnt; ++i) {
        // vertex array
        graph->vertexArray[i] = offset;
        int arc_num = node_matrix[i].size();
        for(int j = 0;j < arc_num; ++j) {
            graph->edgeArray[offset + j] = node_matrix[i][j].id;
            graph->weightArray[offset + j] = node_matrix[i][j].dis;
			if( node_matrix[i][j].dis > max_edge)
				max_edge =  node_matrix[i][j].dis;
        }
        offset += arc_num;
    }
	printf("Max weight: %d\n", max_edge);
    delete []node_matrix;
}

int getSourceVertices(int **sourceVertices, char *sourceFile) {
    int sourceCount = 0;
    std::vector<int> src_node;
    std::ifstream infile;
	infile.open(sourceFile, std::ios::in);
	if (!infile.is_open()) {
        printf("Can not open the ss file!\n");
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

    *sourceVertices = (int *)malloc(sizeof(int) * sourceCount);
    std::copy(src_node.begin(), src_node.end(), *sourceVertices);
    return sourceCount;
}

void releaseGraph(GraphData *graph) {
    free(graph->vertexArray);
    free(graph->edgeArray);
    free(graph->weightArray);
}

int main(int argc, char **argv) {
    if(argc != 4) {
        printf("Usage: ./main [graph file path] [source node file path] [result file path]\n");
        exit(1);
    }
    char graphFile[50];
    memcpy(graphFile, argv[1], strlen(argv[1]));
    graphFile[strlen(argv[1])] = '\0';
    char srcFile[50];
    memcpy(srcFile, argv[2], strlen(argv[2]));
    srcFile[strlen(argv[2])] = '\0';

    cl_platform_id platform;
    cl_context gpuContext;
    cl_context cpuContext;
    cl_int errNum;
    clock_t t;

    cl_uint numPlatforms;
    errNum = clGetPlatformIDs(1, &platform, &numPlatforms);
    if(errNum != CL_SUCCESS || numPlatforms <= 0) {
        printf("Failed to find any OpenCL platforms.\n");
        exit(1);
    }

    // create the OpenCL context on GPU devices
    cl_context_properties properties[3];
    // context properties list -- must be terminated with 0
    properties[0] = CL_CONTEXT_PLATFORM;    // specifies the platform to use
    properties[1] = (cl_context_properties) platform;
    properties[2] = 0;
    gpuContext = clCreateContextFromType(properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
    if(errNum != CL_SUCCESS) {
        printf("No GPU devices found.\n");
        exit(1);            
    }

    // create the OpenCL context on GPU devices
    cpuContext = clCreateContextFromType(0, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
    if(errNum != CL_SUCCESS) {
        printf("No CPU devices found.\n");
        // exit(1);
    }

    t = clock();
    GraphData graph;
    buildGraph(&graph, graphFile);
    printf("Building graph time: %.2f\n", (clock() - t) * 1.0 / CLOCKS_PER_SEC);

    printf("Vertex Count: %d\n", graph.vertexCount);
    printf("Edge Count: %d\n", graph.edgeCount);

    int sourceNum;
    int *sourceVertices;
    sourceNum = getSourceVertices(&sourceVertices, srcFile);
<<<<<<< HEAD
    prinf("Source count: %d\n", sourceNum);
=======
	printf("Source count: %d\n", sourceNum);
>>>>>>> ad168b7dba88f056e9b290d0cc84db80c9b94934
    assert(sourceNum > 0);

    long *results = (long *)malloc(sizeof(long) * sourceNum * graph.vertexCount);
    t = clock();
    // runDijkstraMultiGPUandCPU(gpuContext, cpuContext, &graph, sourceVertices,
    //                                                   results, sourceNum);
    runDijkstraMultiGPU(gpuContext, &graph, sourceVertices, results, sourceNum);
    printf("Processing time: %.2f\n", (clock() - t) * 1.0 / CLOCKS_PER_SEC / sourceNum);
    
    std::ofstream result_file;
    char resFile[50];
	memcpy(resFile, argv[3], strlen(argv[3]));
	resFile[strlen(argv[3])] = '\0'; 
    result_file.open(resFile, std::ios::out | std::ios::trunc);
	int offset = 0;
    for(int i = 0;i < sourceNum; ++i) {
		long dis = 0;
		int cur_src = sourceVertices[i];
		assert(results[offset + cur_src] == 0);
		for(int j = 0; j < graph.vertexCount; ++j) {
			assert(results[offset + j] >= 0);
			if(results[offset + j] == 2147483647) {
				printf("Wow!\n");
				continue;
			}
			dis += results[offset + j];
		}
		offset += graph.vertexCount;
		result_file << "ss " << dis << std::endl;
    }
	result_file.close();
<<<<<<< HEAD
    assert(offset == sourceNum * graph.vertexCount);
=======
	assert(offset == sourceNum * graph.vertexCount);
>>>>>>> ad168b7dba88f056e9b290d0cc84db80c9b94934

    releaseGraph(&graph);
    free(sourceVertices);
    free(results);
}
