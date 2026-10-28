import torch


def try_gpu(i=0, output=False):
    """
    Return gpu(i) if exists, otherwise return cpu(). This is mostly taken from d2l.ai
    :param i: Which GPU to use
    :param output: Whether to output GPU statistics
    :return: The GPU device
    """
    if torch.cuda.device_count() >= i + 1:
        name = torch.cuda.get_device_name(i)
        if output:
            print(f"GPU Name: {name}")
            print('Memory Usage:')
            print('\tTotal:\t\t', round(torch.cuda.get_device_properties(i).total_memory / 1024 ** 3, 1), 'GB')
            print('\tAllocated:\t', round(torch.cuda.memory_allocated(i) / 1024 ** 3, 1), 'GB')
            print('\tCached:\t\t', round(torch.cuda.memory_reserved(i) / 1024 ** 3, 1), 'GB\n')
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
