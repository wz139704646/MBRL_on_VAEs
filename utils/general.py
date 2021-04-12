# general utils
def merge_dicts(dicts, merge_fn):
    """merge the dicts (all dicts have the same keys) according to specific merge function
    :param dicts: a list of dict
    :param merge_fn: a function indicating how to merge the list of values of the same key
    """
    new_dict = {k: [dic[k] for dic in dicts] for k in dicts[0]}
    new_dict = {k: merge_fn(v) for k, v in new_dict.items()}
    return new_dict


def integrate_dicts(dicts):
    """integrate the dicts, replace the value for the same key with the later one
    :param dicts: a list of dict
    :return: a merged dict
    """
    new_dict = {}
    for d in dicts:
        new_dict.update(d)

    return new_dict
