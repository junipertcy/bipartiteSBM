""" i/o utilities """
import numpy as np


def get_edgelist(f_edgelist, delimiter=None):
    """
    This function returns an edgelist list from a file.

    Parameters
    ----------
    f_edgelist : ``str``
        The path to the edgelist text file.

    delimiter : ``str``
        The delimiter that separate the edges.

    Returns
    -------
    edgelist : :class:`numpy.ndarray`
        The numpy list of tupled edges.
    """
    edgelist = []
    with open(f_edgelist, "r") as f:
        for line in f:
            line = line.replace('\r', '').replace('\n', '')  # remove all line breaks!
            if delimiter is not None:
                edge = line.split(delimiter)
                edgelist.append((int(edge[0]), int(edge[1])))
            else:
                try:
                    edge = line.split(' ')
                    edgelist.append((int(edge[0]), int(edge[1])))
                except ValueError:
                    try:
                        edge = line.split('\t')
                        edgelist.append((int(edge[0]), int(edge[1])))
                    except ValueError:
                        try:
                            edge = line.split(',')
                            edgelist.append((int(edge[0]), int(edge[1])))
                        except ValueError:
                            raise ValueError("Tried delimiters ' ', '\t', and ',', but none work...")

    return np.array(edgelist, dtype=np.int_)


def get_types(f_types):
    """
    This function returns an edgelist list from a file.

    Parameters
    ----------
    f_edgelist : ``str``
        The path to the types file

    Returns
    -------
    edgelist : ``list``
        The list of types of each node.

    Examples
    --------
    >>> from biSBM.ioutils import get_types
    >>> edgelist = get_types(f_types)
    >>> print(edgelist)

    """
    types = []
    with open(f_types, "r") as f:
        for line in f:
            types.append(int(line.replace('\n', "")))

    return np.array(types, dtype=np.int_)


def save_mb_to_file(path, mb):
    """Save the group membership list to a file path.

    Parameters
    ----------
    path : ``str``, required
        File path for the list to save to.

    mb : ``list[int]``, required
        Group membership list.

    """
    assert type(mb) is list, "[ERROR] the type of the second input parameter should be a list"
    num_nodes = len(mb)
    with open(path, "w") as f:
        for i in range(0, num_nodes):
            f.write(str(mb[i]) + "\n")
