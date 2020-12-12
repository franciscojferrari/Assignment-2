""" This file is created as a template for question 2.4 in DD2434 - Assignment 2.

    We encourage you to keep the function templates as is.
    However this is not a "must" and you can code however you like.
    You can write helper functions however you want.

    You do not have to implement the code for finding a maximum spanning tree from scratch. We provided two different
    implementations of Kruskal's algorithm and modified them to return maximum spanning trees as well as the minimum
    spanning trees. However, it will be beneficial for you to try and implement it. You can also use another
    implementation of maximum spanning tree algorithm, just do not forget to reference the source (both in your code
    and in your report)! Previously, other students used NetworkX package to work with trees and graphs, keep in mind.

    We also provided an example regarding the Robinson-Foulds metric (see Phylogeny.py).

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py file),
    and modify them as needed. In addition to the sample files given to you, it is very important for you to test your
    algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format.

    Note that the sample files are tab delimited with binary values (0 or 1) in it.
    Each row corresponds to a different sample, ranging from 0, ..., N-1
    and each column corresponds to a vertex from 0, ..., V-1 where vertex 0 is the root.
    Example file format (with 5 samples and 4 nodes):
    1   0   1   0
    1   0   1   0
    1   0   0   0
    0   0   1   1
    0   0   1   1

    Also, I am aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). I wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    For this assignment, we gave you a single tree mixture (q2_4_tree_mixture).
    The mixture has 3 clusters, 5 nodes and 100 samples.
    We want you to run your EM algorithm and compare the real and inferred results
    in terms of Robinson-Foulds metric and the likelihoods.
    """
from math import isnan
import numpy as np
import matplotlib.pyplot as plt
from Tree import TreeMixture, Node, Tree
import sys
from Kruskal_v1 import Graph
import dendropy
import pandas as pd

min_float = sys.float_info.min


def save_results(loglikelihood, topology_array, theta_array, filename):
    """ This function saves the log-likelihood vs iteration values,
        the final tree structure and theta array to corresponding numpy arrays. """

    likelihood_filename = filename + "_em_loglikelihood.npy"
    topology_array_filename = filename + "_em_topology.npy"
    theta_array_filename = filename + "_em_theta.npy"
    print("Saving log-likelihood to ", likelihood_filename, ", topology_array to: ", topology_array_filename,
          ", theta_array to: ", theta_array_filename, "...")
    np.save(likelihood_filename, loglikelihood)
    np.save(topology_array_filename, topology_array)
    np.save(theta_array_filename, theta_array)


def compute_likehood(treeMixture, samples, num_samples):
    clusters = treeMixture.clusters
    res = np.ones((num_samples, len(clusters)))
    pi = treeMixture.pi

    for n, x in enumerate(samples):
        for k, t in enumerate(clusters):
            topology = t.get_topology_array()
            theta = t.get_theta_array()
            # we visit each x sample
            for i, sample in enumerate(x):
                parent_node = topology[i]
                # root node
                if isnan(parent_node):
                    res[n, k] *= theta[0][sample]
                else:
                    CPD = theta[i]
                    res[n, k] *= CPD[x[int(parent_node)]][sample]
    res += min_float
    loglikehood = np.sum(np.log(np.sum(res * pi, axis=1)))
    return loglikehood


def compute_responsabilities(treeMixture, samples, num_samples):
    # print("COMPUTE Responsabilities \n")
    pi = treeMixture.pi

    clusters = treeMixture.clusters
    res = np.ones((num_samples, len(clusters)))

    for n, x in enumerate(samples):
        for k, t in enumerate(clusters):
            topology = t.get_topology_array()
            theta = t.get_theta_array()
            # we visit each x sample
            for i, sample in enumerate(x):
                parent_node = topology[i]
                # root node
                if isnan(parent_node):
                    res[n, k] *= theta[0][sample]
                else:
                    CPD = theta[i]
                    res[n, k] *= CPD[x[int(parent_node)]][sample]
    res += min_float

    loglikehood = np.sum(np.log(np.sum(res * pi, axis=1)))

    res = pi * (res / np.tile(np.sum(res * pi,
                                     axis=1).reshape((num_samples, 1)), len(clusters)))
    return loglikehood, res


def calculate_qkab(r, samples, num_clusters):
    # print("COMPUTE qkab \n")
    num_samples, num_nodes = samples.shape

    r_sum = np.sum(r, axis=0)
    a_ = b_ = 2
    qkab = np.zeros((num_nodes, num_nodes, a_, b_, num_clusters))

    for s, t in np.ndindex(num_nodes, num_nodes):
        for a, b in np.ndindex(a_, b_):

            same = [i for i, ab in enumerate(
                samples[:, (s, t)]) if (ab == [a, b]).all()]
            sum_r_ab = np.sum(r[same], axis=0)
            # print(sum_r_ab)
            qkab[s, t, a, b] = (sum_r_ab / r_sum) + min_float

    return qkab


def calculate_theta():
    pass


def calculate_qka_qkb(r, samples, num_clusters):
    # print("COMPUTE qka_qkb( \n")

    num_samples, num_nodes = samples.shape

    r_sum = np.sum(r, axis=0)
    qka_kb = np.zeros((num_nodes, 2, num_clusters))

    for s, a in np.ndindex(num_nodes, 2):
        sum_r_a = np.sum(r[np.where(samples[:, s] == a)], axis=0)
        qka_kb[s, a] = (sum_r_a / r_sum) + min_float
    return qka_kb


def calculate_Iqk(qkab, qka_kb):
    # print("COMPUTE Iqk( \n")
    s_t, a_b, k = qka_kb.shape
    Iqk = np.zeros((s_t, s_t, k))

    for s, t in np.ndindex(s_t, s_t):
        for a, b in np.ndindex(a_b, a_b):
            Iqk[s, t] += qkab[s, t, a, b] * \
                np.log(qkab[s, t, a, b] /
                       ((qka_kb[s, a] * qka_kb[t, b]) + min_float))
    return Iqk


def calculate_max_trees(Iqk, qkab, qka_kb, sample, r):
    # print("COMPUTE MaxTrees \n")]
    new_clusters = []

    s_t, k = Iqk.shape[1:]
    for k in range(k):
        graph = Graph(s_t)
        for s in range(s_t):
            for t in range(s + 1, s_t):
                graph.addEdge(s, t, Iqk[s, t, k])

        mst = graph.maximum_spanning_tree()
        edges = np.array(mst)[:, 0:2]
        topology = np.ones(s_t)
        topology[0] = np.nan
        vnodes = [0]

        while len(vnodes) > 0:
            cnode = vnodes[0]
            same = np.transpose(np.stack(np.where(edges == cnode)))
            vnodes = vnodes[1:]
            for i in same:
                child_node = int(edges[i[0], 1-i[1]])
                topology[child_node] = cnode
                vnodes.append(child_node)
            if np.size(same) > 0:
                edges = np.delete(edges, same[:, 0], axis=0)

        # New CPD of the root node --> we can reuse the QKA
        cat = [qka_kb[0, :, k]]

        # CPD for the rest of the nodes --> we need to calculate the ML for the child - father nodes
        rk = r[:, k]
        for j in range(1, sample.shape[1]):
            k_theta = np.zeros((2, 2))
            parent_t = topology[j]
            for a, b in np.ndindex(2, 2):
                parent_selected = np.where(sample[:, int(parent_t)] == a)[0]
                child_selected = np.where(sample[:, j] == b)[0]
                parent_child_selected = list(
                    set(parent_selected).intersection(set(child_selected)))
                k_theta[a][b] = np.sum(
                    rk[parent_child_selected])/np.sum(rk[parent_selected]) + min_float
            k_theta = k_theta / \
                np.tile(np.sum(k_theta, axis=1).reshape((2, 1)), 2)
            cat.append([k_theta[0, :], k_theta[1, :]])

        new_tree = Tree()
        new_tree.load_tree_from_direct_arrays(topology, cat)
        new_tree.alpha = [1.0] * 2
        new_clusters.append(new_tree)
    return new_clusters


def compute_em(treeMixture, num_clusters, samples, max_num_iter=100):

    # print("COMPUTE EM \n")
    num_samples, num_nodes = samples.shape
    logLikehoods = []
    # treeMixture.print()
    pi = np.ones((1, num_clusters))
    pi = pi / np.sum(pi)
    treeMixture.pi = pi

    for _ in range(max_num_iter):

        loglikehood, r = compute_responsabilities(
            treeMixture, samples, num_samples)

        logLikehoods.append(loglikehood)

        # Update categorical distribution
        treeMixture.pi[0] = np.mean(r, axis=0)

        # Construct directed graphs
        qkab = calculate_qkab(r, samples, num_clusters)
        qka_qkb = calculate_qka_qkb(r, samples, num_clusters)
        Iqk = calculate_Iqk(qkab, qka_qkb)
        treeMixture.clusters = calculate_max_trees(
            Iqk, qkab, qka_qkb, samples, r)
    # print(logLikehoods)
    return logLikehoods, treeMixture


def em_algorithm(seed_val, samples, num_clusters, max_num_iter=100, n_sieving=10):
    """
    This function is for the EM algorithm.
    :param seed_val: Seed value for reproducibility. Type: int
    :param samples: Observed x values. Type: numpy array. Dimensions: (num_samples, num_nodes)
    :param num_clusters: Number of clusters. Type: int
    :param max_num_iter: Maximum number of EM iterations. Type: int
    :return: loglikelihood: Array of log-likelihood of each EM iteration. Type: numpy array.
                Dimensions: (num_iterations, ) Note: num_iterations does not have to be equal to max_num_iter.
    :return: topology_list: A list of tree topologies. Type: numpy array. Dimensions: (num_clusters, num_nodes)
    :return: theta_list: A list of tree CPDs. Type: numpy array. Dimensions: (num_clusters, num_nodes, 2)

    This is a suggested template. Feel free to code however you want.
    """

    # Set the seed
    np.random.seed(seed_val)

    # Compute the sieving
    sievings = np.random.randint(0, 10000, n_sieving)
    top_logLikehoods = []
    num_samples, num_nodes = samples.shape
    treeMixtures = []
    # loglikelihood = []
    print("Running Sievings...")
    for sieving in sievings:
        np.random.seed(sieving)
        tm = TreeMixture(num_clusters=num_clusters, num_nodes=num_nodes)
        tm.simulate_pi(seed_val=sieving)
        tm.simulate_trees(seed_val=sieving)
        tm.sample_mixtures(num_samples=num_samples, seed_val=sieving)

        loglikelihood_ = compute_likehood(tm, samples, num_samples)
        top_logLikehoods.append(loglikelihood_)

        treeMixtures.append(tm)

    print("Running EM algorithm...")
    top_logLikehood_index = top_logLikehoods.index(max(top_logLikehoods))

    np.random.seed(sievings[top_logLikehood_index])
    tm = treeMixtures[top_logLikehood_index]

    # tm = TreeMixture(num_clusters=num_clusters, num_nodes=num_nodes)
    # tm.simulate_pi(seed_val=seed_val)
    # tm.simulate_trees(seed_val=seed_val)
    # tm.sample_mixtures(num_samples=num_samples, seed_val=seed_val)

    loglikelihood, tree_mix = compute_em(
        tm, num_clusters, samples, max_num_iter)

    print("EM algorithm finished...")

    theta_list = []
    topology_list = []

    for tree in tree_mix.clusters:
        topology_list.append(tree.get_topology_array())
        theta_list.append(tree.get_theta_array())

    loglikelihood = np.array(loglikelihood)
    topology_list = np.array(topology_list)
    theta_list = np.array(theta_list)

    return loglikelihood, topology_list, theta_list, tree_mix


def main():
    print("Hello World!")
    print("This file demonstrates the flow of function templates of question 2.4.")

    seed_val = 123

    sample_filename = "data/q2_4/q2_4_tree_mixture.pkl_samples.txt"
    output_filename = "q2_4_results.txt"
    real_values_filename = "data/q2_4/q2_4_tree_mixture.pkl"
    num_clusters = 3

    print("\n1. Load samples from txt file.\n")

    samples = np.loadtxt(sample_filename, delimiter="\t", dtype=np.int32)
    num_samples, num_nodes = samples.shape
    # print("\tnum_samples: ", num_samples, "\tnum_nodes: ", num_nodes)
    # print("\tSamples: \n", samples)

    print("\n2. Run EM Algorithm.\n")

    loglikelihood, topology_array, theta_array, infered_tree_mix = em_algorithm(
        seed_val, samples, num_clusters=num_clusters, max_num_iter=100)

    print("\n3. Save, print and plot the results.\n")

    save_results(loglikelihood, topology_array, theta_array, output_filename)

    for i in range(num_clusters):
        print("\n\tCluster: ", i)
        print("\tTopology: \t", topology_array[i])
        print("\tTheta: \t", theta_array[i])

    plt.figure(figsize=(8, 3))
    plt.subplot(121)
    plt.plot(np.exp(loglikelihood), label='Estimated')
    plt.ylabel("Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.subplot(122)
    plt.plot(loglikelihood, label='Estimated')
    plt.ylabel("Log-Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.legend(loc=(1.04, 0))
    plt.show()

    if real_values_filename != "":
        print("\n4. Retrieve real results and compare.\n")
        print("\tComparing the results with real values...")

        real_trees = TreeMixture(0, 0)
        real_trees.load_mixture(real_values_filename)

        print("\t4.1. Make the Robinson-Foulds distance analysis.\n")
        den_tns = dendropy.TaxonNamespace()
        distance_metrics = {"Real Tree": [],
                            "Infered Tree": [], "RF Distance": []}
        for i, rtc in enumerate(real_trees.clusters):
            for j, itc in enumerate(infered_tree_mix.clusters):
                rt_den = dendropy.Tree.get(
                    data=rtc.newick, schema="newick", taxon_namespace=den_tns)
                it_den = dendropy.Tree.get(
                    data=itc.newick, schema="newick", taxon_namespace=den_tns)
                distance_metrics["Real Tree"].append(i)
                distance_metrics["Infered Tree"].append(j)
                distance_metrics["RF Distance"].append(
                    dendropy.calculate.treecompare.symmetric_difference(rt_den, it_den))
        distance_df = pd.DataFrame.from_dict(distance_metrics)
        print(distance_df)
        print("\t4.2. Make the likelihood comparison.\n")
        log_likehood_real_tree = compute_likehood(
            real_trees, samples, num_samples)
        print(f"log likehood of real tree mixture {log_likehood_real_tree}")
        print(f"log likehood of infered tree mixture {loglikelihood[-1]}")


if __name__ == "__main__":
    main()
