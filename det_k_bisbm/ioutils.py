""" i/o utilities """


from retrying import retry


def get_edgelist(f_edgelist, delimiter=","):
    """
        This function returns an edgelist list from a file.

        Parameters
        ----------
        f_edgelist : str
            The path to the edgelist file

        delimiter : str
            The delimiter in the file

        Returns
        -------
        edgelist : list
            The list of tupled edges.

        >>> import ioutils
        >>> edgelist = get_edgelist(f_edgelist, ",")
        >>> print(edgelist)

    """
    import re
    edgelist = []
    with open(f_edgelist, "rb") as f:
        for line in f:
            line = line.replace('\r', '').replace('\n', '')  # remove all line breaks!
            edge = re.split(delimiter, line)
            # edgelist.append((str(int(edge[0]) - 1), str(int(edge[1]) - 1)))
            edgelist.append((str(int(edge[0])), str(int(edge[1]))))
    f.close()
    return edgelist

def get_types(f_types):
    """
        This function returns an edgelist list from a file.

        Parameters
        ----------
        f_edgelist : str
            The path to the types file

        Returns
        -------
        edgelist : list
            The list of types of each node.

        >>> import ioutils
        >>> edgelist = get_types(f_types)
        >>> print(edgelist)

    """
    types = []
    with open(f_types, "rb") as f:
        for line in f:
            types.append(str(int(line.replace('n', ""))))
    f.close()
    return types


@retry(stop_max_attempt_number=10, wait_fixed=60000)  # wait 60 seconds
def open_biDCSBMcomms_file(num_sweep_):
    '''
        :return: file handle
    '''
    f = open(
        self.f_kl_output + '/biDCSBMcomms' +
        str(int(num_sweep_)) + '.tsv', 'rb'
    )
    return f

@retry(stop_max_attempt_number=10, wait_fixed=60000)  # wait 60 seconds
def get_bisbm_score_file(f_kl_output, num_sweep_):
    '''
        :return: file handle
    '''
    f = open(
        f_kl_output + '/biDCSBMcomms' + str(int(num_sweep_)) + '.score', 'rb'
    )
    return f

def get_score_by_index(num_sweep_):
    f = get_bisbm_score_file(num_sweep_)
    for ind, line in enumerate(f):
        score = float(line.split('\n')[0])
    f.close()
    return score

def get_of_group_by_index(num_sweep_):
    of_group = []
    f = open_biDCSBMcomms_file(num_sweep_)
    for ind, line in enumerate(f):
        of_group.append(int(line.split('\n')[0]))
    f.close()
    return of_group




# TODO: make this a command line tool; re-write the hard f_config setting
def load_conf(config, subset=None):
    import os
    import configobj

    f_config = config
    # Raise Error if config file not found
    if not os.path.exists(f_config):
        msg = "{} not found".format(f_config)
        raise IOError(msg)

    config = configobj.ConfigObj(f_config)

    if subset is None:
        return config

    # Load all the global keys
    output = configobj.ConfigObj()

    for key, val in config.items():
        if type(val) is not configobj.Section:
            output[key] = val

    # Add in the local subset information
    output.update(config[subset])

    return output
