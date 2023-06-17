import numpy as np
import wget
import zipfile
from numpy import genfromtxt
import random


def str_list_to_float(str_list):
    """
    Converts a list of strings to a list of floats.

    Args:
        str_list: List of strings

    Returns:
        List of floats
    """
    return [float(item) for item in str_list]


def str_list_to_int(str_list):
    """
    Converts a list of strings to a list of integers.

    Args:
        str_list: List of strings

    Returns:
        List of integers
    """
    return [int(item) for item in str_list]


def read_edges(train_filename, test_filename):
    """read data from files

    Args:
        train_filename: training file name
        test_filename: test file name

    Returns:
        node_num: int, number of nodes in the graph
        graph: dict, node_id -> list of neighbors in the graph
    """

    graph = {}
    nodes = set()
    train_edges = read_edges_from_file(train_filename)
    test_edges = read_edges_from_file(
        test_filename) if test_filename != "" else []

    for edge in train_edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
        if graph.get(edge[0]) is None:
            graph[edge[0]] = []
        if graph.get(edge[1]) is None:
            graph[edge[1]] = []
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])

    for edge in test_edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
        if graph.get(edge[0]) is None:
            graph[edge[0]] = []
        if graph.get(edge[1]) is None:
            graph[edge[1]] = []

    return len(nodes), graph


def read_edges_from_file(filename):
    """
    Reads edges from a file and returns them as a list of lists.

    Args:
        filename: File name

    Returns:
        List of edges, where each edge is a list [node_id_1, node_id_2]
    """
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [str_list_to_int(line.split()) for line in lines]
    return edges


def read_embeddings(filename, n_node, n_embed):
    """
    Reads pretrained node embeddings.

    Args:
        filename: File name of the embeddings
        n_node: Number of nodes in the graph
        n_embed: Number of embedding dimensions

    Returns:
        Numpy array representing the embeddings
    """

    with open(filename, "r") as f:
        lines = f.readlines()[1:]  # skip the first line
        embedding_matrix = np.random.rand(n_node, n_embed)
        for line in lines:
            emd = line.split()
            embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
        return embedding_matrix


def reindex_node_id(edges):
    """reindex the original node ID to [0, node_num)

    Args:
        edges: list, element is also a list like [node_id_1, node_id_2]
    Returns:
        new_edges: list[[1,2],[2,3]]
        new_nodes: list [1,2,3]
    """

    node_set = set()
    for edge in edges:
        node_set = node_set.union(set(edge))

    node_set = list(node_set)
    new_nodes = set()
    new_edges = []
    for edge in edges:
        new_edges.append([node_set.index(edge[0]), node_set.index(edge[1])])
        new_nodes = new_nodes.add(node_set.index(edge[0]))
        new_nodes = new_nodes.add(node_set.index(edge[1]))

    new_nodes = list(new_nodes)
    return new_edges, new_nodes


def generate_neg_links(train_filename, test_filename, test_neg_filename):
    """
    generate neg links for link prediction evaluation
    Args:
        train_filename: the training edges
        test_filename: the test edges
        test_neg_filename: the negative edges for test
    """

    train_edges = read_edges_from_file(train_filename)
    test_edges = read_edges_from_file(test_filename)
    neighbors = {}  # dict, node_ID -> list_of_neighbors
    for edge in train_edges + test_edges:
        if neighbors.get(edge[0]) is None:
            neighbors[edge[0]] = []
        if neighbors.get(edge[1]) is None:
            neighbors[edge[1]] = []
        neighbors[edge[0]].append(edge[1])
        neighbors[edge[1]].append(edge[0])
    nodes = set([x for x in range(len(neighbors))])

    # for each edge in the test set, sample a negative edge
    neg_edges = []

    for i in range(len(test_edges)):
        edge = test_edges[i]
        start_node = edge[0]
        neg_nodes = list(nodes.difference(set(neighbors[edge[0]] + [edge[0]])))
        neg_node = np.random.choice(neg_nodes, size=1)[0]
        neg_edges.append([start_node, neg_node])
    neg_edges_str = [str(x[0]) + "\t" + str(x[1]) + "\n" for x in neg_edges]
    with open(test_neg_filename, "w+") as f:
        f.writelines(neg_edges_str)


def softmax(x):
    e_x = np.exp(x - np.max(x))  # for computation stability
    return e_x / e_x.sum()


snapshots = ['https://nih.figshare.com/ndownloader/files/40431644',  # 2023-04
             'https://nih.figshare.com/ndownloader/files/35116603',  # 2022-04
             'https://nih.figshare.com/ndownloader/files/27968064',  # 2021-04
             'https://nih.figshare.com/ndownloader/files/24681569',  # 2020-04
             ]


def download_edges_csv(snapshot_urls):
    """
    Downloads the edge CSV files from the snapshot URLs.

    Args:
        snapshot_urls: List of URLs for downloading the edge CSV files
    """
    for ind, snap in enumerate(snapshots):
        wget.download(snap)
        with zipfile.ZipFile('\open_citation_collection.zip', 'r') as zip:
            zip.extract('open_citation_collection.csv',
                        path='edges_file_'+str(ind), pwd=None)


def load_edges(ind, max_rows=100):
    """
    Loads the edge data from the CSV file.

    Args:
        ind: Index of the snapshot
        max_rows: Maximum number of rows to load

    Returns:
        Numpy array representing the edge data
    """
    my_data = genfromtxt('\edges_file_'+str(ind)+'\open_citation_collection.csv',
                         delimiter=',', max_rows=max_rows)
    return my_data


def str_list_to_int(str_list):
    """
    Converts a list of strings to a list of integers.

    Args:
        str_list: List of strings

    Returns:
        List of integers
    """
    return [int(item) for item in str_list]


def read_edges_from_file(filename):
    """
    Reads the edges from a file and returns them as a list of integer pairs.

    Args:
        filename: Name of the file to read

    Returns:
        List of integer pairs representing the edges
    """
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [str_list_to_int(line.split()) for line in lines]
    return edges


def getDataSet():
    """
    Downloads the edge CSV files from the snapshot URLs, loads the edge data, and returns it.

    Returns:
        Numpy array representing the edge data
    """
    download_edges_csv(snapshots)
    res = load_edges(0, 50000)
    return res


def createTestFiles():
    """
    Creates train and test files by removing a portion of edges from the original train files.
    """
    inds = []
    for i in range(3):
        arr = []
        removed = []
        with open("CA-GrQc_train"+str(i)+".txt", "r") as f:
            for line in f.readlines():
                arr.append(line)

        length = int(len(arr)*0.15)
        for j in range(length):
            if i == 0:
                index = random.randint(0, len(arr))
                inds.append(index)
            else:
                index = inds[j]

            if i == 2:
                removed.append(arr[index])
            arr.pop(index)

        with open("CA-GrQc_train"+str(i)+".txt", "w") as f:
            for line in arr:
                f.write(line)

    # with last snapshot
    with open("CA-GrQc_test.txt", "w") as f:
        for line in removed:
            f.write(line)

    generate_neg_links("CA-GrQc_train2.txt",
                       "CA-GrQc_test.txt", "CA-GrQc_test_neg.txt")


def createSnapshotFiles():
    """
    Creates snapshot files by loading edge data from each snapshot and saving them as text files.
    """
    max_keys = 5000
    max_edges = 50000
    for i in range(3):
        res = load_edges(i, max_edges+1)  # load 50000 edged from each snapshot
        snap_map = {}  # {key:[,]}
        for j in range(1, max_edges+1):
            keys = snap_map.keys()
            if(str(int(res[j][1])) in snap_map):  # was not an int
                snap_map[str(int(res[j][1]))] += [str(int(res[j][0]))]
            if(len(keys) != max_keys):
                snap_map[str(int(res[j][1]))] = [str(int(res[j][0]))]

        with open("/content/snapshot"+str(i)+".txt", "w+") as f:
            for key, values in snap_map.items():
                for value in values:
                    line = f"{key} {value}\n"
                    f.write(line)


def remap_edges(edges, number):
    """
    Remaps the node IDs in the edge list to consecutive integers starting from 0.

    Args:
        edges: List of edges
        number: Number identifier for the remapped snapshot file

    Writes:
        Remapped edge list to a text file
    """
    x = set([row[0] for row in edges])
    y = set([row[1] for row in edges])
    nodes = list(x.union(y))
    print(len(nodes))

    i = 0
    map = {}
    for i in range(len(nodes)):
        map[nodes[i]] = i

    for i in range(len(edges)):
        edges[i][0] = map[edges[i][0]]
        edges[i][1] = map[edges[i][1]]

    with open("/content/snapshotNew"+str(number)+".txt", "w+") as f:
        for line in edges:
            f.write(str(line[0])+" "+str(line[1])+"\n")


def remap_edges_from_files():
    """
    Remaps the node IDs in the snapshot files to consecutive integers starting from 0.
    """
    for i in range(3):
        my_data = []
        with open("snapshot"+str(i)+".txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                value = []
                x, y = line.split()
                value.append(x)
                value.append(y)
                my_data.append(value)
        remap_edges(my_data, i)


def create_adjacency_matrix(edge_list):
    """
    Creates an adjacency matrix from the given edge list.

    Args:
        edge_list: List of edges

    Returns:
        Adjacency matrix as a list of lists
    """
    print(edge_list[1])

    max_id = max(max(edge) for edge in edge_list)
    max_id = int(max_id)  # Convert max_id to an integer
    adj_matrix = [[0 for j in range(max_id)] for i in range(max_id)]

    for edge in edge_list:
        i, j = edge
        adj_matrix[i-1][j-1] = 1
    for row in adj_matrix:
        print(row)
    return adj_matrix
