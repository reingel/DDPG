import unittest
from replaybuffer import Transition, rb

trans0 = Transition(1., 5., -1., 2.)
trans1 = Transition(2., 4., -2., 3.)

rb.push(trans0)
rb.push(trans1)

class TestReplayBuffer(unittest.TestCase):
    def test_push(self):
        self.assertEqual(rb.buffer, [trans0, trans1])
    
    def test_sample(self):
        self.assertIn(*rb.sample(1), rb.buffer)

if __name__ == '__main__':
    unittest.main()