# coding = utf-8 
# python 3.6.7 
# Created by wuyang at2019/4/10

from node2vec import Node2Vec


def node2vec_X(graph, node2vec_node_X_path):
    # import networkx as nx
    # graph_path = "F:\Windows10\Desktop\ENZYMES_g118\ENZYMES_g118.edges"

    # Create a graph
    # graph = nx.Graph()
    # graph.add_node(1)
    # graph.add_edges_from(example_testing.get_edges_list(graph_path))
    # graph = nx.karate_club_graph()

    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)  # Use temp_folder for big graphs

    # Embed nodes
    model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

    # Look for most similar nodes
    model.wv.most_similar('2')  # Output node names are always strings

    # Save embeddings for later use
    model.wv.save_word2vec_format(node2vec_node_X_path)

    # Save model for later use
    model.save("F:\Windows10\Desktop\KARATE\KARATE_g_model.txt")

    # Embed edges using Hadamard method
    from node2vec.edges import HadamardEmbedder

    edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

    # Look for embeddings on the fly - here we pass normal tuples
    edges_embs[('1', '2')]
    ''' OUTPUT
    array([ 5.75068220e-03, -1.10937878e-02,  3.76693785e-01,  2.69105062e-02,
           ... ... ....
           ..................................................................],
          dtype=float32)
    '''

    # Get all edges in a separate KeyedVectors instance - use with caution could be huge for big networks
    edges_kv = edges_embs.as_keyed_vectors()

    # Look for most similar edges - this time tuples must be sorted and as str
    edges_kv.most_similar(str(('1', '2')))

    # Save embeddings for later use
    edges_kv.save_word2vec_format("F:\Windows10\Desktop\KARATE\KARATE_g_edge.txt")
