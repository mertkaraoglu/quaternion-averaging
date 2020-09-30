# Mert Asim Karaoglu, 2020
# Heavily inspired by Dr. Tolga Birdal's work: https://github.com/tolgabirdal/averaging_quaternions
# Based on 
# Markley, F. Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman. 
# "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30, 
# no. 4 (2007): 1193-1197.
import torch


def quaternion_average(a: torch.Tensor) -> torch.Tensor:
    r"""Quaternion average based on Markley, F. Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman. "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30, no. 4 (2007): 1193-1197.

    Args:
        a: N x 4 tensor each row representing a different data point, assumed to represent a unit-quaternion vector; i.e. [x, y, z, w]

    Returns:
        torch.Tensor: N x 4 tensor each row representing a different data point, represents a unit-quaternion vector; i.e. [x, y, z, w]


    """
    # handle the antipodal configuration
    a[a[:, 3] < 0] = -1 * a[a[:, 3] < 0]

    a = a.view(-1, 4, 1)

    eigen_values, eigen_vectors = torch.matmul(a, a.transpose(1, 2)).mean(dim=0).eig(True)

    return eigen_vectors[:, eigen_values.argmax(0)[0]].view(1, 4)


def weighted_quaternion_average(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    r"""Weighted quaternion average based on Markley, F. Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman. "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30, no. 4 (2007): 1193-1197.

    Args:
        a: N x 4 tensor each row representing a different data point, assumed to represent a unit-quaternion vector; i.e. [x, y, z, w]
        w: N x 1 tensor each row representing a different float for weight

    Returns:
        torch.Tensor: N x 4 tensor each row representing a different data point, represents a unit-quaternion vector; i.e. [x, y, z, w]


    """
    # handle the antipodal configuration
    a[a[:, 3] < 0] = -1 * a[a[:, 3] < 0]

    a = a.view(-1, 4, 1)

    eigen_values, eigen_vectors = torch.matmul(a.mul(w.view(-1, 1, 1)), a.transpose(1, 2)).sum(dim=0).div(w.sum()).eig(
        True)

    return eigen_vectors[:, eigen_values.argmax(0)[0]].view(1, 4)