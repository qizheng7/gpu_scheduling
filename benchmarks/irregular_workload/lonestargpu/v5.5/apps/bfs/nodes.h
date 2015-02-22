struct Node
{
  unsigned node;      // node to work on 
  unsigned start;     // starting edge number
  unsigned length;    // end edge number
//  Node *next;
};

struct Node_threads
{
  unsigned total_nodes;
  Node nodes[MAX_NODES];
};

struct Node_graph
{
//  unsigned node;       // node to work on 
  unsigned length;     // number of rest edges 
  unsigned start;      // start edge that has not been grabbed 
};

struct head_Node
{
  Node *next;
  int total_length;   //only used by head node
  int total_nodes;    //only used by head node
};
