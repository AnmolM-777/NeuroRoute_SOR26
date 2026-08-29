"""
Unit tests for NetworkRoutingEnv Gymnasium environment.
"""

import unittest
import numpy as np
import gymnasium as gym

from neuroroute.ai.env import NetworkRoutingEnv
from neuroroute.router.plane import RouterNode


class TestNetworkRoutingEnv(unittest.TestCase):

    def test_env_initialization(self):
        """Test space definitions, observation shape, and basic initialization."""
        env = NetworkRoutingEnv(num_nodes=5, max_queue_capacity=100)

        # Action space
        self.assertEqual(env.action_space.n, 5)
        self.assertIsInstance(env.action_space, gym.spaces.Discrete)

        # Observation space: 3 * num_nodes + 1 = 16
        self.assertEqual(env.observation_space.shape, (16,))
        self.assertIsInstance(env.observation_space, gym.spaces.Box)
        self.assertEqual(env.observation_space.dtype, np.float32)

        env.close()

    def test_action_masking(self):
        """Test action mask generation for valid links."""
        env = NetworkRoutingEnv(num_nodes=5)
        mask = env.get_action_mask(0)

        self.assertEqual(len(mask), 5)
        self.assertEqual(mask.dtype, bool)
        # Default ring topology: node 0 connects to 1 and 4, plus cross link to 2
        self.assertTrue(mask[1])
        self.assertTrue(mask[4])
        self.assertTrue(mask[2])
        self.assertFalse(mask[0])  # No self loop

    def test_reset_behavior(self):
        """Test reset with seeds and custom options."""
        env = NetworkRoutingEnv(num_nodes=5)
        obs, info = env.reset(
            seed=42,
            options={"current_node": 1, "destination_node": 3, "queue_depths": [10, 20, 30, 40, 50]},
        )

        self.assertEqual(obs.shape, (16,))
        self.assertEqual(info["current_node"], 1)
        self.assertEqual(info["destination_node"], 3)
        self.assertEqual(env.step_count, 0)
        self.assertEqual(env.dropped_packets, 0)
        self.assertEqual(env.successful_deliveries, 0)

        env.close()

    def test_step_valid_forwarding(self):
        """Test valid step execution between connected nodes."""
        env = NetworkRoutingEnv(num_nodes=5)
        env.reset(options={"current_node": 0, "destination_node": 3, "queue_depths": [0, 10, 0, 0, 0]})

        # Action 1 (node 1 is connected to node 0 in ring topology)
        obs, reward, terminated, truncated, info = env.step(1)

        self.assertEqual(info["current_node"], 1)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertLess(reward, 0.0)  # Step penalty

        env.close()

    def test_step_invalid_action_penalty(self):
        """Test invalid action penalty when choosing a non-existent link."""
        # Create topology with strictly disconnected nodes
        adj_matrix = np.zeros((3, 3), dtype=np.float32)
        adj_matrix[0, 1] = 10.0  # Link only 0 -> 1

        env = NetworkRoutingEnv(num_nodes=3, topology_graph=adj_matrix)
        env.reset(options={"current_node": 0, "destination_node": 1})

        # Node 0 -> Node 2 is not connected!
        obs, reward, terminated, truncated, info = env.step(2)

        self.assertEqual(reward, -100.0)  # Heavy invalid action penalty
        self.assertEqual(env.dropped_packets, 1)
        self.assertIn("error", info)

        env.close()

    def test_destination_arrival_reward(self):
        """Test positive reward and termination upon reaching target node."""
        env = NetworkRoutingEnv(num_nodes=5)
        env.reset(options={"current_node": 0, "destination_node": 1, "queue_depths": [0, 0, 0, 0, 0]})

        # Action 1 reaches destination node 1
        obs, reward, terminated, truncated, info = env.step(1)

        self.assertEqual(reward, 98.0)  # max(10, 100 - 2*1) = 98
        self.assertTrue(terminated)
        self.assertEqual(env.successful_deliveries, 1)

        env.close()

    def test_queue_overflow_penalty(self):
        """Test heavy penalty when stepping to a full buffer node."""
        env = NetworkRoutingEnv(num_nodes=5, max_queue_capacity=100)
        # Node 1 queue is full (100)
        env.reset(options={"current_node": 0, "destination_node": 4, "queue_depths": [0, 100, 0, 0, 0]})

        obs, reward, terminated, truncated, info = env.step(1)

        self.assertEqual(reward, -50.0)
        self.assertFalse(terminated)
        self.assertEqual(env.dropped_packets, 1)

        env.close()

    def test_data_plane_queue_sync(self):
        """Test syncing node queue lengths from RouterNode data plane instances."""
        env = NetworkRoutingEnv(num_nodes=3)
        env.reset()

        nodes = [RouterNode("0", buffer_size=10), RouterNode("1", buffer_size=10), RouterNode("2", buffer_size=10)]
        # Add a packet manually to node 1's queue
        nodes[1]._queue.put_nowait("dummy_packet")

        env.sync_from_data_plane(nodes)
        self.assertEqual(env.queue_depths[1], 1.0)

        env.close()

    def test_render_ansi(self):
        """Test rendering output in ANSI mode."""
        env = NetworkRoutingEnv(num_nodes=3, render_mode="ansi")
        env.reset()
        output = env.render()

        self.assertIsInstance(output, str)
        self.assertIn("Node: 0", output)

        env.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)