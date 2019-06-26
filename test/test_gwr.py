import numpy as np
import tempfile
import unittest

from gwr import GWR


class TestGwr(unittest.TestCase):
    def test_gwr(self):
        np.random.seed(42)

        size1 = 100
        means1 = [0, 0]
        cov1 = np.eye(2)
        g1 = np.random.multivariate_normal(means1, cov1, size=size1)

        size2 = 100
        means2 = [5, 5]
        cov2 = np.eye(2)
        g2 = np.random.multivariate_normal(means2, cov2, size=size2)

        dataset = np.vstack([g1, g2])
        gwr = GWR()
        gwr.fit(dataset, normalize=True)
        # self.assertEqual(len(conns), len(set(conns)))

        # print(len(gwr.nodes), len(gwr.conns))

        self.assertLess(len(gwr.nodes), size1 + size2)
        self.assertLess(len(gwr.conns), (size1 + size2) * (size1 + size2 - 1) / 2)

        with tempfile.NamedTemporaryFile() as tmpfile:
            gwr.dump(tmpfile.name)
            gwr2 = GWR.load(tmpfile.name)

            self.assertEqual(len(gwr.conns), len(gwr2.conns))
            self.assertEqual(len(gwr.nodes), len(gwr2.nodes))
            self.assertTrue(all([(n1 == n2) for n1, n2 in zip(gwr.nodes, gwr2.nodes)]))

            for c1 in gwr.conns:
                self.assertEqual(gwr.conns[c1], gwr2.conns[c1])

    def test_gwr_continue(self):
        np.random.seed(42)

        size1 = 100
        means1 = [0, 0]
        cov1 = np.eye(2)
        g1 = np.random.multivariate_normal(means1, cov1, size=size1)

        size2 = 1000
        means2 = [5, 5]
        cov2 = np.eye(2)
        g2 = np.random.multivariate_normal(means2, cov2, size=size2)

        dataset = np.vstack([g1, g2])
        gwr = GWR()
        gwr.fit(dataset, normalize=True, iters=1)
        gwr.fit(dataset, normalize=True, iters=1)

        node_pts = gwr.get_weights()
        delta_ws = gwr.delta_ws
        act_thr = gwr.activity_thr
        num_nodes = len(gwr.nodes)
        num_conns = len(gwr.conns)
        self.assertLess(act_thr, 0.95)   # activity threshold should be adjusted during the training
        self.assertLess(num_nodes, 180)
        self.assertLess(num_conns, 290)
        self.assertLess(num_nodes, num_conns)
        self.assertEqual(len(delta_ws), 2)

        gwr = GWR()
        gwr.fit(dataset, normalize=True, iters=2)
        self.assertEqual(gwr.activity_thr, act_thr)   # activity threshold should be adjusted during the training
        self.assertEqual(len(gwr.nodes), num_nodes)
        self.assertEqual(len(gwr.conns), num_conns)
        self.assertEqual(len(gwr.delta_ws), len(delta_ws))
        self.assertTrue(np.all(np.array(gwr.get_weights()) - np.array(node_pts) == 0))
        self.assertTrue(np.all(np.array(gwr.delta_ws) - np.array(delta_ws) == 0))

    def test_4points(self):
        dataset = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])

        gwr = GWR(firing_counter=0.1)
        gwr.fit(dataset, normalize=False, iters=1000)
        self.assertEqual(len(gwr.nodes), 4)
        # print([min([np.linalg.norm(x-node.w) for node in gwr.nodes]) for x in dataset])
        self.assertTrue(all([min([np.linalg.norm(x-node.w) for node in gwr.nodes]) < 0.092 for x in dataset]))
