import unittest
import torch
import quaternion_averaging


class QuaternionAveragingTest(unittest.TestCase):

    def test_quaternion_averaging(self):

        print("Running simple quaternion averaging tests.")
        # Test the simple averaging
        orientations = torch.tensor([[0.3061862, 0.1767767, 0.3061862, 0.8838835],
                                     [0.5915064, 0.1584936, 0.591506, 0.5245191]])
        ground_truth = torch.tensor([0.4662, 0.1741, 0.4662, 0.7314])
        output = quaternion_averaging.quaternion_average(orientations)
        equal = torch.allclose(output.flatten(), ground_truth.flatten(), atol=1e-04)
        self.assertEqual(True, equal, "Quaternion Averaging Test: simple averaging test failed!")

        # Test the antipodal condition
        orientations = torch.tensor([[-0.3061862, -0.1767767, -0.3061862, -0.8838835],
                                     [0.5915064, 0.1584936, 0.591506, 0.5245191]])
        ground_truth = torch.tensor([0.4662, 0.1741, 0.4662, 0.7314])
        output = quaternion_averaging.quaternion_average(orientations)
        equal = torch.allclose(output.flatten(), ground_truth.flatten(), atol=1e-04)
        self.assertEqual(True, equal, "Quaternion Averaging Test: simple averaging antipodality test failed!")


    def test_weighted_quaternion_averaging(self):

        print("Running weighted quaternion averaging tests.")
        # Test the weighted averaging
        orientations = torch.tensor([[0.3061862, 0.1767767, 0.3061862, 0.8838835],
                                     [0.5915064, 0.1584936, 0.591506, 0.5245191]])
        weights = torch.tensor([[2], [1]])
        ground_truth = torch.tensor([0.4111, 0.1766, 0.4111, 0.7943])
        output = quaternion_averaging.weighted_quaternion_average(orientations, weights)
        equal = torch.allclose(output.flatten(), ground_truth.flatten(), atol=1e-04)
        self.assertEqual(True, equal, "Quaternion Averaging Test: weighted averaging test failed!")

        # Test the weighted antipodal condition
        orientations = torch.tensor([[-0.3061862, -0.1767767, -0.3061862, -0.8838835],
                                     [0.5915064, 0.1584936, 0.591506, 0.5245191]])
        weights = torch.tensor([[2], [1]])
        ground_truth = torch.tensor([0.4111, 0.1766, 0.4111, 0.7943])
        output = quaternion_averaging.weighted_quaternion_average(orientations, weights)
        equal = torch.allclose(output.flatten(), ground_truth.flatten(), atol=1e-04)
        self.assertEqual(True, equal, "Quaternion Averaging Test: weighted averaging antipodality test failed!")

if __name__ == '__main__':
    unittest.main()
