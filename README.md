# SSSP
For 2019 Parallel Course Project

## Serial Implementation
Use Dijkastra algorithm with Fibonacci heap.

| Dataset | NY | NE | CTR |
|---|---|---|---|
| Serial | 1.51s | 10.05s | 124.65s |

## OpenCL Implementation Without Fiboheap
Compile method: 
```
gcc -o main -I$ATISTREAMSDKROOT/include -L$ATISTREAMSDKROOT/lib/x86_64 main.cpp Dijkstra.cpp -lOpenCL -lstdc++ -lpthread
```

Result:

| Dataset | NY | NE | CTR |
|---|---|---|---|
| Serial | 0.27s | 2.37s | 88.50s |



## Reference

- The fibonacci heap/queue is copied from [repo](https://github.com/beniz/fiboheap.git).
- The dataset can be downloaded from [here](http://users.diag.uniroma1.it/challenge9/competition.shtml)
