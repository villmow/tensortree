# tensortree 
Work with trees just like arrays.

## Concept

A tree is stored in a **pre-order traversal** of its nodes in three arrays of the same length:
 1. A tensor of `parent` pointers. This tensor contains the index of the parent a node at index i and defines 
    the structure of the tree. Root should have the parent -1: `parents = [-1, 0, 1, 1, 0, 5]`.
 2. A list or tensor of `node data` (for example a token in NLP). This can be a list of anything or a 
    torch tensor. If it is a tensor, pytorch operations will be used.
 3. a tensor of `descendants`, in which the number of descendants of a node is stored: `descendants = [5, 2, 0, 0, 1, 0]`. 
 This will be computed from `parents` array, if it is not passed during construction or vice versa.

Note: If possible every array will be converted to pytorch tensors. 

Operations that modify the structure of the tree (for example by adding or removing nodes) are expensive and
will always return a new `TensorTree` object with new and copied data (new storage). This should be reduced 
to a minimum.

# Usage

Define a tree with an array of parents:
```python
>>> import tensortree
>>> parents = [-1, 0, 1, 1, 0, 4]
>>> tree = tensortree.tree(parents)
```
and print a string version of it using `tree.pprint()`:
```
TensorTree():
 ╰── 0. 0
    ├── 1. 1
    │   ├── 2. 2
    │   ╰── 3. 3
    ╰── 4. 4
        ╰── 5. 5
```

Per node an array with data (preferably a tensor) can be passed. Here a list of dictionaries is used, along 
with the parents array from above:
```python
>>> node_data = [
    {"name": "A", "some_attribute": False},
    {"name": "B", "some_attribute": False},
    {"name": "C", "some_attribute": False},
    {"name": "D", "some_attribute": False},
    {"name": "E", "some_attribute": False},
    {"name": "F", "some_attribute": False},
]
>>> tree = tensortree.tree(parents=parents, node_data=node_data)
>>> tree.pprint()
TensorTree():
 ╰── 0. {'name': 'A', 'some_attribute': False}
    ├── 1. {'name': 'B', 'some_attribute': False}
    │   ├── 2. {'name': 'C', 'some_attribute': False}
    │   ╰── 3. {'name': 'D', 'some_attribute': False}
    ╰── 4. {'name': 'E', 'some_attribute': False}
        ╰── 5. {'name': 'F', 'some_attribute': False}

# render node data
>>> tree.pprint(node_renderer=lambda x: x["name"])
TensorTree():
 ╰── 0. A
    ├── 1. B
    │   ├── 2. C
    │   ╰── 3. D
    ╰── 4. E
        ╰── 5. F
```
The `pprint` and `pformat` methods take a node renderer as an argument, which is a 
callable, that receives the data of a node (from the given array) and returns a string. 
Additionally the `max_nodes` argument can be used to restrict the amount of nodes printed.

### Navigation

Each node is identified by a `node_idx`: The index of the node in the original array (recap that this needs
to be sorted pre-order depth first). Most functions of a `TensorTree` return indices of nodes. Either the index 
of a single node, a tensor of indices (which can be used for advanced indexing like in pytorch tensors), or a boolean
mask to index a tensor. Sometimes an iterable over indices is returned (if the indices can't be computed
directly).

You can navigate inside the tree using the `node_idx` of a node. To select subtree as a view (still located in the bigger tree, 
with the same storage) simply use the `__getitem__` notation along with a node: 

```python
# same tree as above
>>> subtree = tree[1]
>>> subtree.pprint()
TensorTree():
 ╰── 1. {'name': 'B', 'some_attribute': False}
    ╰── 2. {'name': 'C', 'some_attribute': False}
    ╰── 3. {'name': 'D', 'some_attribute': False}
>>> root = subtree[None]    # different view
>>> root == tree
False
>>> root.data = tree.data  # same storage
True
```

You can retrieve the three arrays (`node_data`, `descendants`, `parents`) for a specific view of the tree:
```python

```

## Usage

You can retrieve a nodes parent by using `tree.get_parent(node_idx)`.

