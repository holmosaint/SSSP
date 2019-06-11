#include<iostream>
#include<sstream>
#include<fstream>
#include<string>
#include<ctime>
#include<vector>
#include<deque>
#include<set>
#include<cassert>
#include<algorithm>
#include"fiboheap/fiboqueue.h"
#include"fiboheap/fiboheap.h"

#define INF 1e15

using namespace std;

struct node_struct {
	int id;
	int dis;
	node_struct(int _id, int _dis) :id(_id), dis(_dis) {}
};

long long dijkastra(int src, int node_num, deque<node_struct> *node_matrix) {
	long long dis_sum = 0;
	vector<int> finish_set;
	finish_set.push_back(src);
	long long *dis_matrix = new long long[node_num];
	for (int i = 0; i < node_num; ++i)
		dis_matrix[i] = -1;
	dis_matrix[src] = 0;
	set<int> pending_node, done_node;
	done_node.insert(src);

	int lst_node = src;
	while (finish_set.size() != node_num) {
		int finish_cnt = finish_set.size();
		// update dis set
		long long lst_dis = dis_matrix[lst_node];
		for (deque<node_struct>::iterator it = node_matrix[lst_node].begin(); it != node_matrix[lst_node].end(); ++it) {
			int node_id = it->id;
			long long dis = (long long)it->dis;
			if (done_node.find(node_id) == done_node.end()) {
				pending_node.insert(node_id);
			}
			if (dis_matrix[node_id] < 0)
				dis_matrix[node_id] = dis;
			else dis_matrix[node_id] = min(dis_matrix[node_id], dis + lst_dis);
		}
		// find the minimize dis node in pending_node
		long long min_dis = 1e15;
		set<int>::iterator min_it = pending_node.end();
		for (set<int>::iterator it = pending_node.begin(); it != pending_node.end(); ++it) {
			int node_id = *it;
			int tmp_dis = dis_matrix[node_id];
			if (tmp_dis < min_dis) {
				min_dis = tmp_dis;
				min_it = it;
			}
		}
		dis_sum += min_dis;
		lst_node = *min_it;
		done_node.insert(*min_it);
		pending_node.erase(min_it);
		
	}

	delete[]dis_matrix;
	return dis_sum;
}

long long dijkastra_queue(int src, int node_num, deque<node_struct> *node_matrix) {
    long long dis_sum = 0;
    set<int> finish_set;
    FibQueue<long long> fib_queue;
    deque<FibHeap<long long>::FibNode *> node_queue;

    // initialize
    long long val;
    for(int i = 0;i < node_num; ++i) {
        if(i == src)
            val = 0;
        else val = INF;
        FibHeap<long long>::FibNode *x = fib_queue.push(val);
        x->id = i;
        node_queue.push_back(x);
    }
    
    FibHeap<long long>::FibNode * min_node;
    int min_id, lst_cnt;
    long long min_dis;
    printf("Src %d\n", src);
    while(finish_set.size() < node_num) {
        // printf("Finish cnt: %d\n", finish_set.size());
        min_node = fib_queue.extract_min();
        // fib_queue.pop();
        min_dis = min_node->key;
        min_id = min_node->id;
        finish_set.insert(min_id);
        dis_sum += min_dis;
        // printf("Finish cnt: %d\n", finish_set.size());
        // printf("Delete node: %d with dis %lld\n", min_id, min_dis);
        for(deque<node_struct>::iterator it = node_matrix[min_id].begin(); it != node_matrix[min_id].end(); ++it) {
            int node_id = it->id;
            // printf("Processing node %d\n", node_id);
            if(finish_set.find(node_id) != finish_set.end())
                continue;
            FibHeap<long long>::FibNode *cur_node = node_queue[node_id];
            // printf("Processing node %d\n", node_id);
            long long new_dis = min_dis + it->dis;
            if(new_dis < cur_node->key) {
                // printf("Processing node %d\n", node_id);
                // printf("Old dis: %lld; New dis: %lld\n", cur_node->key, new_dis);
                // printf("Node addr: 0x%x\n", cur_node);
                fib_queue.decrease_key(cur_node, new_dis); 
                // printf("After decrease dis: %lld\n", cur_node->key);
            }
        }
    }

    return dis_sum;
}
int main() {
	ifstream infile;
	clock_t t = clock();
	infile.open("./data/USA-road-d.NY.gr", ios::in);
	if (!infile.is_open()) {
		cout << "Can not open the gr file!" << endl;
		// system("pause");
		// exit(1);
		return 1;
	}
	printf("Open file time: %.2fs\n", (clock() - t) * 1.0 / CLOCKS_PER_SEC);
	
	t = clock();
	std::string line;
	int v_cnt, arc_cnt;
	deque<node_struct> *node_matrix = NULL;
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
	printf("Build graph time: %.2fs\n", (clock() - t) * 1.0 / CLOCKS_PER_SEC);

	vector<int> src_node;
	int src_cnt;
	infile.open("./data/USA-road-d.NY.ss", ios::in);
	if (!infile.is_open()) {
		cout << "Can not open the ss file!" << endl;
		// system("pause");
		// exit(1);
		return 1;
	}
	while (getline(infile, line)) {
		if (line[0] == 'c')
			continue;
		stringstream lineStream(line);
		string type, tmp;
		int src;
		lineStream >> type;
		if (type == "p") {
			lineStream >> tmp >> tmp >> tmp;
			lineStream >> src_cnt;
		}
		else if (type == "s"){
			lineStream >> src;
            // printf("%d\n", src);
            // exit(1);
			src_node.push_back(src - 1);
		}
	}
	assert(src_cnt == src_node.size());
    infile.close();

    t = clock();    
    ofstream result_file;
    result_file.open("./result.NY.txt", ios::out | ios::trunc);
	for (unsigned int i = 0; i < src_node.size(); ++i) {
        // long long dis_sum = dijkastra(src_node[i], v_cnt, node_matrix);
        // long long dis_sum = dijkastra_heap(src_node[i], v_cnt, node_matrix);
        long long dis_sum = dijkastra_queue(src_node[i], v_cnt, node_matrix);
		printf("The dis sum from src %d is %lld\n", src_node[i], dis_sum);
        result_file << "ss " << dis_sum << endl;
        // break;
	}
    printf("Average processing time: %.2fs\n", (clock() - t) * 1.0 / CLOCKS_PER_SEC / src_node.size());
    
    result_file.close();
	delete[]node_matrix;

	// system("pause");
	return 0;
}
