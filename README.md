# SSSP
For 2019 Parallel Course Project

## Serial Implementation
Use Dijkastra algorithm with Fibonacci heap.

| Dataset | NY | NE | CTR |
|---|---|---|---|
| Serial | 1.51s | 10.05s | 124.65s |
| Serial Total w/o output| 1149.78 | 1179.25s | 1611.45 |
| Serial Total w/ output| 1277.68 | 1255.44s | 1789.86 |

## OpenCL Implementation Without Fiboheap
Compile method: 
```
gcc -o main -I$ATISTREAMSDKROOT/include -L$ATISTREAMSDKROOT/lib/x86_64 main.cpp Dijkstra.cpp -lOpenCL -lstdc++ -lpthread
```

Result:

| Dataset | NY | NE | CTR |
|---|---|---|---|
| Parallel Avg. | 0.27s | 2.37s | 88.50s |
| Parallel Total w/o Output| 145.29s | 226.47 | 2119.96 |
| Parallel Total w/ Output| 145.29s | 227.06 | 2163.52 |



## Reference

- The fibonacci heap/queue is copied from [repo](https://github.com/beniz/fiboheap.git).
- The dataset can be downloaded from [here](http://users.diag.uniroma1.it/challenge9/competition.shtml)
