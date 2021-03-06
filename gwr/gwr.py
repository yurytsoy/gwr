import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

import numpy as np
import os


class GWR:
    def __init__(self, activity_thr=0.99, firing_counter=0.3, epsn=0.01, epsb=0.1, max_edge_age=5, random_state=42):
        self.nodes = None
        self.conns = dict()
        self.activity_thr = activity_thr
        self.firing_counter = firing_counter or np.inf
        self.random_state = random_state
        self.eps_n = epsn
        self.eps_b = epsb
        self.max_edge_age = max_edge_age
        self.delta_ws = None
        self.mins = None
        self.deltas = None

        self.winner_firing_counters = np.array([1-(1-np.exp(-1.05*k/3.33))/1.05 for k in range(100)])
        self.neighbor_firing_counters = np.array([1-(1-np.exp(-1.05*k/14.3))/1.05 for k in range(100)])

    def _get_next_winner_firing_counter(self, cur_value):
        if cur_value == 0:
            return self.winner_firing_counters[0]
        if cur_value > self.winner_firing_counters[-1]:
            return self.winner_firing_counters[self.winner_firing_counters < cur_value][0]
        return self.winner_firing_counters[-1]

    def _get_next_neighbor_firing_counter(self, cur_value):
        if cur_value == 0:
            return self.neighbor_firing_counters[0]
        if cur_value > self.neighbor_firing_counters[-1]:
            return self.neighbor_firing_counters[self.neighbor_firing_counters < cur_value][0]
        return self.neighbor_firing_counters[-1]

    def _init(self, arg_xs, normalize):
        if normalize:
            self.mins = np.min(arg_xs, axis=0)
            self.deltas = np.max(arg_xs, axis=0) - self.mins

        self.nodes = []
        for k in range(2):
            random_idxs = np.unique(np.random.choice(len(arg_xs), int(len(arg_xs) / 4)))
            sub_xs = arg_xs[random_idxs, :]
            node_w = np.mean(sub_xs, axis=0)
            if normalize:
                node_w = (node_w - self.mins) / self.deltas
            self.nodes.append(Node(id=k, w=node_w))
        self.conns = dict()

    def fit(self, arg_xs, y=None, normalize=True, iters=None, verbose=False, delta_thr=None, warm_start=False):
        np.random.seed(self.random_state)

        if iters is None:
            iters = 1

        if delta_thr is None:
            delta_thr = 1e-4

        if not warm_start or self.nodes is None:
            self._init(arg_xs, normalize)

        xs = arg_xs
        if self.mins is not None and self.deltas is not None:
            xs = (arg_xs - self.mins) / self.deltas

        deltas = []

        if verbose:
            print("delta thr:", delta_thr)
            print("activity thr:", self.activity_thr)
        for t in range(iters):
            if len(self.nodes) > 0.1*len(xs) and self.activity_thr > 0.8:
                self.activity_thr *= 0.95
                self.activity_thr = max(self.activity_thr, 0.8)

            cur_deltas = []
            for x in xs:
                cur_dw = self._fit_sample(x)
                cur_deltas.append(np.sqrt(np.dot(cur_dw, cur_dw) / len(cur_dw)))
            if len(cur_deltas) > 0:
                deltas.append(np.mean(cur_deltas))

            if verbose:
                print("{:3}:".format(t+1), len(cur_deltas), deltas[-1], len(self.nodes), len(self.conns))
            if deltas[-1] < delta_thr:
                break
        if self.delta_ws is None:
            self.delta_ws = deltas
        else:
            self.delta_ws.extend(deltas)

    def _fit_sample(self, x):
        """

        :param x:
        :return:
        """
        def _get_next_node_id():
            return 1+max([n.id for n in self.nodes])

        # find the best and second best matching nodes.
        node1, node2 = self._find_best_matching_nodes(x)

        cur_conn = (node1.id, node2.id) if node1.id < node2.id else (node2.id, node1.id)
        self.conns[cur_conn] = 0

        # add a new node or update node1 and node2.
        activity1 = np.exp(-np.linalg.norm(x - node1.w) / np.sqrt(len(x)))
        if activity1 < self.activity_thr and node1.firing_counter < self.firing_counter and node2.firing_counter < self.firing_counter:
            new_node = Node(id=_get_next_node_id(), w=0.5*(node1.w + x))
            self.nodes.append(new_node)
            self.conns[(node1.id, new_node.id)] = 0
            self.conns[(node2.id, new_node.id)] = 0
            del self.conns[cur_conn]
            dw1 = new_node.w - x
        else:
            dw1 = self.eps_b * node1.firing_counter * (x - node1.w)
            node1.w += dw1
            nnodes = self._find_neighbors(node1)
            for nnode in nnodes:
                dw2 = self.eps_n * nnode.firing_counter * (x - nnode.w)
                nnode.w += dw2

        # age connections incident to node1 and update firing counters of its neighborhs.
        node1.firing_counter = self._get_next_winner_firing_counter(node1.firing_counter) #  h0 - 1/alpha_b * (1 - np.exp(-alpha_b * t / tau_b))
        for conn in self.conns:
            if node1.id not in conn:
                continue

            self.conns[conn] += 1
            neighbor_id = (set(conn) - set([node1.id])).pop()
            neighbor_node = None
            for n in self.nodes:
                if n.id == neighbor_id:
                    neighbor_node = n
                    break
            neighbor_node.firing_counter = self._get_next_neighbor_firing_counter(neighbor_node.firing_counter)   # h0 - 1 / alpha_n * (1 - np.exp(-alpha_n * t / tau_n))

        self._remove_dangling_nodes()
        self._remove_too_old_edges()

        return dw1

    def _remove_dangling_nodes(self):
        node_ids = {node.id: node for node in self.nodes}
        for c in self.conns:
            if c[0] in node_ids:
                del node_ids[c[0]]
            if c[1] in node_ids:
                del node_ids[c[1]]
        for node_id, node in node_ids.items():
            self.nodes.remove(node)

    def _remove_too_old_edges(self):
        conn_ids = list(self.conns)
        for conn_id in conn_ids:
            if self.conns[conn_id] > self.max_edge_age:
                del self.conns[conn_id]

    def _find_best_matching_nodes(self, argx):
        min_dist1, min_dist2 = np.inf, np.inf
        node1, node2 = None, None

        for node in self.nodes:
            dist = np.linalg.norm(argx - node.w)

            if dist < min_dist2:
                min_dist2 = dist
                node2 = node
            if dist < min_dist1:
                node2 = node1
                min_dist2 = min_dist1
                node1 = node
                min_dist1 = dist

        assert(node1.id != node2.id)
        return node1, node2

    def _find_neighbors(self, node):
        ids = []
        for conn in self.conns:
            if conn[0] == node.id:
                ids.append(conn[1])
            if conn[1] == node.id:
                ids.append(conn[0])
        node_ids = {node.id: node for node in self.nodes}
        return [node_ids[node_id] for node_id in ids]

    def fit_predict(self, xs, y=None):
        self.fit(xs, y)
        return self.predict(xs)

    def predict(self, xs):
        pass

    def get_weights(self):
        res = []
        for node in self.nodes:
            if self.mins is not None and self.deltas is not None:
                res.append(node.w * self.deltas + self.mins)
            else:
                res.append(node.w)
        return res

    def dump(self, filename):
        with open(filename, mode='w') as fd:
            fd.write(jsonpickle.dumps(self))

    @staticmethod
    def load(filename):
        with open(filename, mode='r') as fd:
            s = fd.read()
        res = jsonpickle.loads(s)
        if res.conns:
            conns = dict()
            for str_key, value in res.conns.items():
                conns[eval(str_key)] = value
            res.conns = conns
        return res


class Node:
    def __init__(self, id=None, w=None):
        self.id = id
        self.w = w
        self.firing_counter = 1

    def __eq__(self, other):
        if type(other) != type(self):
            return False

        return (other.id == self.id) and np.all(other.w == self.w) and (other.firing_counter == self.firing_counter)
