/////////////////////////////////////
/////////////////////////////////////
//// assgin Nodes to head Nodes only in balanced_nodes
//TODO: not using "edges_needed" even after residues are used
// use linked list pointer to traverse linked list (new version use headNode.total_nodes)
#define BFS_VARIANT "lonestar"
#include "cutil_subset.h"

///////////////////////////////////////////
//caogao
#define WARP_SIZE 32
#define MAX_WARPS 32
#define MAX_GROUPS 64
#define MAX_BLOCKS 64
#include "nodes.h"
//#define DEBUG 1
//#define TEST 1
//#define ONE_WARP 1

__device__
void printout(head_Node *head_nodes, Node *node_warps, const unsigned tid, const unsigned wid){
  Node* node = head_nodes->next; 
////////////////////////
///debug
  if (node == NULL) {
    printf("break\n");
  } else {
////////////////////
    printf("printout: total_length %u\ttotal_nodes %u\tnode_ptr %u\tnode %u\tstart %u\tlength %utid %u\twid %u\n", 
            head_nodes->total_length, head_nodes->total_nodes, node, node->node, node->start, node->length, tid, wid);
    while(node->next != NULL) {
      node = node->next;
      printf("node %u\tstart %u\tlength %u\n", node->node, node->start, node->length);
    };
  }
}

__device__
void next_tid(unsigned *tid, unsigned *residue_left, unsigned *edges_thread) {
  (*tid)++;
  if (*residue_left > 0) {
    (*residue_left)--;
    if (residue_left == 0) (*edges_thread)--;
  }
}

__device__
bool increment(unsigned *ptr, const int bound) {
  (*ptr)++;
  if (*ptr >= bound) {
    printf("max node numbers reached\n");
    return false;
  } 
  return true;
}

__device__
bool balance_nodes(Node_graph *node_graphs, head_Node *head_nodes, Node *node_warps, 
                    const unsigned average, const unsigned residue, const unsigned wid, const unsigned bid) {
  unsigned ptr = 0; // number of Nodes used in *node_warp
//  Node *current_node = (node_warps + bid * blockDim.x + wid * WARO_SIZE + ptr);
  Node *current_node = (node_warps + (bid * blockDim.x / WARP_SIZE + wid) * MAX_GROUPS + ptr);
  unsigned tid = 0; // thread that Node is assigned to
  unsigned node = 0;  // current Node_graph
  unsigned runs = 0;
  unsigned edges_thread = average + 1;            // edges need for each thread. plus 1 because there are residue edges
  unsigned residue_left = residue;
  while (node < WARP_SIZE) {
    //calculate number of the edges we need
    unsigned edges_needed;                        // how many edges left 
    if (head_nodes[tid].next != NULL)  {          // not a new thread 
      edges_needed = edges_thread - head_nodes[tid].total_length;
    } else { 
      edges_needed = edges_thread;     
    }
    if (edges_needed == 0) {
      next_tid(&tid, &residue_left, &edges_thread);
      continue;
    }
#ifdef DEBUG
   printf("bid %u\twid %u\ttid %u\tcurrent node %u\tedges_needed %u\tlength %u\tstart %u\ttotal_length %u\tedges_thread %u\n",
            bid, wid, tid, node, edges_needed, node_graphs[node].length, node_graphs[node].start, 
            head_nodes[tid].total_length, edges_thread
    );
#endif
    //assign the edges
    if (node_graphs[node].length > edges_needed)  { // need to break nodes
      if (!increment(&ptr, MAX_GROUPS)) return false;
      // create new node
      current_node->next = head_nodes[tid].next; 
      current_node->node = node_graphs[node].node;
      current_node->start = node_graphs[node].start;
      current_node->length = edges_needed;
      // insert new node
      head_nodes[tid].next = current_node; 
      head_nodes[tid].total_length += edges_needed; 
      head_nodes[tid].total_nodes++; 
      // remove edges from node_graphs[node]
      node_graphs[node].length -= edges_needed;
      node_graphs[node].start += edges_needed;
      current_node++;  //? TODO
#ifdef DEBUG
      printf("break node %u\tedge_needed %u\tnew_length %u\tnew_start %u\tresidue_left %u\tlength %u\ttid\t%u\n",
            node, edges_needed, node_graphs[node].length, node_graphs[node].start, residue_left,
            head_nodes[tid].total_length, tid
    );
#endif
    } else {  // grab the entire node 
      if (!increment(&ptr, MAX_GROUPS)) return false;
      // create new node
      current_node->next = head_nodes[tid].next; 
      current_node->node = node_graphs[node].node;
      current_node->start = node_graphs[node].start;
      current_node->length = node_graphs[node].length;
      // insert new node
      head_nodes[tid].next = current_node; 
      head_nodes[tid].total_length += node_graphs[node].length; 
      head_nodes[tid].total_nodes++; 
      // move to the next node
      current_node++;
      node++;
#ifdef DEBUG
  if (wid == 0 && bid == 0)
    printf("edges_needed\t%u\tlength\t%u\tnode\t%u\n", edges_needed, node_graphs[node].length, node);
#endif
    }
    if (head_nodes[tid].total_length >= edges_thread) {
      next_tid(&tid, &residue_left, &edges_thread);
#ifdef DEBUG
    printf("node %u\tedge_needed %u\tnew_length %u\tnew_start %u\tresidue_left %u\tlength %u\ttid\t%u\n",
            node, edges_needed, node_graphs[node].length, node_graphs[node].start, residue_left,
            head_nodes[tid].total_length, tid
    );
#endif
    }
    if (runs++ > 256) {printf("max runs reached"); return false;}
  }
  if (wid == 0 || wid == 1)
    for (int iter = 0 ; iter < WARP_SIZE; iter++)  
      printout(&head_nodes[iter], node_warps, iter, wid);
  return true;
}
///////////////////////////////////////////

__global__
void initialize(foru *dist, unsigned int nv) {
	unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
	if (ii < nv) {
		dist[ii] = MYINFINITY;
	}
}

__device__
bool processedge(foru *dist, Graph &graph, unsigned src, unsigned ii, unsigned &dst) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	dst = graph.getDestination(src, ii);
//////////////////////////////////
// caogao
  unsigned wid = threadIdx.x / WARP_SIZE, tid = threadIdx.x % WARP_SIZE, bid = blockIdx.x;
//  if (bid == 0 && wid == 0 && tid == 1) {
//  if (src == 195 || dst == 195) {
//    printf("before: src %u\tdst %u\tedge %u\tdist[src] %u\n", src, dst, ii, dist[src]); 
//    printf("before: src %u\tdst %u\tedge %u\tdist[src] %utid %u\twid %u\tbid %u\n", src, dst, ii, dist[src], tid, wid, bid); 
//  }
//
/////////////////////////////////// 
	if (dst >= graph.nnodes) return 0;

	foru wt = 1;

	foru altdist = dist[src] + wt;
	if (altdist < dist[dst]) {
	 	foru olddist = atomicMin(&dist[dst], altdist);
//////////////////////////////////
// caogao
//  if (bid == 0 && wid == 0 && tid == 1) {
//  if (src == 195 || dst == 195) {
//    printf("after: src %u\tdst %u\tedge %u\tdist[src] %u\n", src, dst, ii, dist[src]); 
//    printf("after: src %u\tdst %u\tedge %u\tdist[src] %utid %u\twid %u\tbid %u\n", src, dst, ii, dist[src], tid, wid, bid); 
//  }
//
/////////////////////////////////// 
		if (altdist < olddist) {
			return true;
		} 
		// someone else updated distance to a lower value.
	}
	return false;
}

__device__
bool processnode(foru *dist, Graph &graph, unsigned work, Node *node_warp) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nn = work;
	if (nn >= graph.nnodes) return 0;
	bool changed = false;
	
	unsigned neighborsize = graph.getOutDegree(nn);
/////////////////////////////////////////////
///caogao
  __shared__ unsigned average[MAX_WARPS];                     // average number of edges
  __shared__ unsigned residue[MAX_WARPS];                     // residue = total - average * WARP_SIZE 
  __shared__ Node_graph node_graphs[MAX_WARPS][WARP_SIZE]; 
  __shared__ head_Node head_node[MAX_WARPS][WARP_SIZE]; 
  unsigned wid = threadIdx.x / WARP_SIZE, tid = threadIdx.x % WARP_SIZE, bid = blockIdx.x;
  __syncthreads();
//  (head_node + bid * blockDim.x + wid * WARP_SIZE + tid)->total_length = 0;
//  (node_warp + bid * blockDim.x + wid * WARP_SIZE + tid)->node = 0;
  if (bid < MAX_GROUPS) {
    node_graphs[wid][tid].node = nn;
    node_graphs[wid][tid].length = neighborsize;
    node_graphs[wid][tid].start = 0;
    head_node[wid][tid].next = NULL;
    head_node[wid][tid].total_length  = 0;
    head_node[wid][tid].total_nodes = 0;
#ifdef DEBUG
  if (bid == 0 && wid == 1)
    printf("tid\t%u\tnode\t%u\tneighborsize\t%u\n", tid, node_graphs[wid][tid].node, node_graphs[wid][tid].length);
#endif
    __syncthreads();
    if (tid == 0) {
      int total = 0;
      for (int i = 0; i < WARP_SIZE; i++)
        total += node_graphs[wid][i].length;
      average[wid] = total / WARP_SIZE;
      residue[wid] = total % WARP_SIZE;
#ifdef ONE_WARP
  #ifdef TEST
    if (bid == 100 && 
        (wid == 0 || wid == 1) )
  #else
    if (bid == 0 && 
        (wid == 0 || wid == 1) )
  #endif
    { 
//      printf("balance_nodes\t%u\t%u\t%u\t%u\n", average[wid], residue[wid], bid, wid);
      balance_nodes(node_graphs[wid], head_node[wid], node_warp, average[wid], residue[wid], wid, bid);
    }
#else
      balance_nodes(node_graphs[wid], head_node[wid], node_warp, average[wid], residue[wid], wid, bid);
#endif
    }
  } else {
    printf("bid bigger than MAX_GROUPS\n");
  }
  __syncthreads();
/////////////////////////////////////////////
//caogao
#ifdef ONE_WARP
  #ifdef TEST
    if (bid == 100 &&
       (wid == 0 || wid == 1) )
  #else
    if (bid == 0 &&
       (wid == 0 || wid == 1) )
  #endif
#else
    if (true)
#endif 
  {
    unsigned node_cnt = head_node[wid][tid].total_nodes; 
    if (node_cnt == 0) return changed;
    Node* node = head_node[wid][tid].next; 
//      printf("printout: thread %u\ttotal_length %u\ttotal_nodes %u\tnode %u\tstart %u\tlength %u\n", 
//              iter, head_nodes[iter].total_length, head_nodes[iter].total_nodes, node->node, node->start, node->length);
  while (true) {
    for (unsigned ii = node->start; ii < (node->start + node->length); ++ii) {
      unsigned dst = graph.nnodes;
      foru olddist = processedge(dist, graph, node->node, ii, dst);
//  #ifdef DEBUG
      if (bid == 8 && wid == 0 && ((tid == 0) || (tid == 1))) 
        printf("balanced: processedge:%u\t%u\t%u\n", bid, wid, tid);
//  #endif
      if (olddist) {
        changed = true;
      }
    }
    node_cnt--;
    if (node_cnt == 0) break;
    node = node->next;
    if (node == NULL) {
//#ifdef DEBUG
      printf("tid %u\twid %u\tnode NULL error!\n", tid, wid); 
      printout(&head_node[wid][tid], node_warp, tid, wid);
//#endif
      break;
    }
//    printf("node %u\tstart %u\tlength %u\n", node->node, node->start, node->length);
  };
  } else {
    for (unsigned ii = 0; ii < neighborsize; ++ii) {
      unsigned dst = graph.nnodes;
      foru olddist = processedge(dist, graph, nn, ii, dst);
//#ifdef DEBUG
    if (bid == 8 && wid == 0 && ((tid == 0) || (tid == 1))) 
      printf("imbalanced: processedge:%u\t%u\t%u\n", bid, wid, tid);
//#endif
      if (olddist) {
        changed = true;
      }
    }
  }
////////////////////////////
	return changed;
}

__global__
void drelax(foru *dist, Graph graph, bool *changed, Node *node_warp) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned start = id * (MAXBLOCKSIZE / blockDim.x), end = (id + 1) * (MAXBLOCKSIZE / blockDim.x);
	for (unsigned ii = start; ii < end; ++ii) {
		if (processnode(dist, graph, ii, node_warp)) {
			*changed = true;
		}
	}
}


void bfs(Graph &graph, foru *dist)
{
	cudaFuncSetCacheConfig(drelax, cudaFuncCachePreferShared);
	foru foruzero = 0.0;
	KernelConfig kconf;
	double starttime, endtime;
	bool *changed, hchanged;
	int iteration = 0;

	kconf.setProblemSize(graph.nnodes);

	kconf.setMaxThreadsPerBlock();
/////////////////////////////////
///caogao
  extern void write_solution(const char *fname, Graph &graph, foru *dist);
//  kconf.setNumberOfBlocks(kconf.getNumberOfBlocks() * 2);
//  kconf.setNumberOfBlockThreads(kconf.getNumberOfBlockThreads() / 2);
	printf("Blocks%u\n", kconf.getNumberOfBlocks());
  printf("BlockThreads%u\n", kconf.getNumberOfBlockThreads());
///////////////////////////////////////
	printf("initializing.\n");
	assert(kconf.coversProblem(0));
	initialize <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph.nnodes);

	CudaTest("initializing failed");

	cudaMemcpy(&dist[0], &foruzero, sizeof(foruzero), cudaMemcpyHostToDevice);

	if (cudaMalloc((void **)&changed, sizeof(bool)) != cudaSuccess) CudaTest("allocating changed failed");

	printf("solving.\n");
	starttime = rtclock();

////////////////////////////
// caogao
//  Node_graph node_graph[MAX_WARPS][WARP_SIZE]; 
//  Node node_warp[MAX_WARPS][WARP_SIZE][2]; 
//  head_Node *head_node;
  Node *node_warp; 

 //cudaDeviceSetLimit(cudaLimitMallocHeapSize, MAX_BLOCKS * MAX_WARPS * MAX_GROUPS * sizeof(Node) * 2);
	if (cudaMalloc((void **)&node_warp, MAX_BLOCKS * MAX_WARPS * MAX_GROUPS * sizeof(Node)) != cudaSuccess) CudaTest("allocating changed failed");
  printf("size:%ld\tstart_address%d\n", MAX_BLOCKS * MAX_WARPS * MAX_GROUPS * sizeof(Node), node_warp);
//////////////////////////
	do {
		++iteration;
		hchanged = false;

		CUDA_SAFE_CALL(cudaMemcpy(changed, &hchanged, sizeof(hchanged), cudaMemcpyHostToDevice));

//////////////////////////
// caogao
//		drelax <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph, changed);

/*    if (kconf.getNumberOfBlocks() > MAX_BLOCKS)  {
      printf("block number exceed MAXBLOCKS\n");
      return;
    }*/
//		drelax <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph, changed, node_warp);
		drelax <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph, changed, node_warp);
//////////////////////////
		CudaTest("solving failed");
////////////////////////////
    printf("iteration:%d\n", iteration);
    switch (iteration) {
    case 1:
	    write_solution("bfs-output_1.txt", graph, dist);
      break;
    case 2:
	    write_solution("bfs-output_2.txt", graph, dist);
      break;
    case 3:
	    write_solution("bfs-output_3.txt", graph, dist);
      break;
    case 4:
	    write_solution("bfs-output_4.txt", graph, dist);
      break;
    case 5:
	    write_solution("bfs-output_5.txt", graph, dist);
      break;
    case 6:
	    write_solution("bfs-output_6.txt", graph, dist);
      break;
    case 7:
	    write_solution("bfs-output_7.txt", graph, dist);
      break;
    default:
	    write_solution("bfs-output.txt", graph, dist);
      break;
    }
////////////////////////////
		CUDA_SAFE_CALL(cudaMemcpy(&hchanged, changed, sizeof(hchanged), cudaMemcpyDeviceToHost));
	} while (hchanged);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock();
	
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, 1000 * (endtime - starttime));

}
