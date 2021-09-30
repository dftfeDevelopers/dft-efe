
def getValidSystems(key):
    if key.lower() not in ["cpu", "gpu", "any"]:
        msg = '''The tag passed should be one of: 'cpu', 'gpu', or
        'any'. Tag passed is {}'''.format(key)
        raise ValueError(msg)

    else:
        if key.lower() == "cpu":
            return [r'.*:.*cpu.*']

        elif key.lower() == "gpu":
            return [r'.*:.*gpu.*']

        else:
            return [r'.*:.*']

def setResources(archTag = 'any', time_limit = "00:02:00", num_nodes = 1, num_tasks_per_node = 1, mem_per_cpu =
                 '5gb', gpus_per_node = 1):
    if archTag.lower() not in ["cpu", "gpu", "any"]:
        msg = '''The tag passed should be one of: 'cpu', 'gpu', or
        'any'. Tag passed is {}'''.format(archTag)
        raise ValueError(msg)

    returnVal = {}
    if archTag.lower() == 'cpu':
        returnVal['cpu'] = {
            'time_limit': time_limit,
            'num_nodes': num_nodes,
            'num_tasks_per_node': num_tasks_per_node,
            'mem_per_cpu': mem_per_cpu
        }

    if archTag.lower() == 'gpu':
        returnVal['gpu'] = {
            'time_limit': time_limit,
            'num_nodes': num_nodes,
            'num_tasks_per_node': num_tasks_per_node,
            'mem_per_cpu': mem_per_cpu,
            'gpus_per_node': gpus_per_node
        }

    if archTag.lower() == 'any':
        returnVal['cpu'] = {
            'time_limit': time_limit,
            'num_nodes': num_nodes,
            'num_tasks_per_node': num_tasks_per_node,
            'mem_per_cpu': mem_per_cpu
        }
        returnVal['gpu'] = {
            'time_limit': time_limit,
            'num_nodes': num_nodes,
            'num_tasks_per_node': num_tasks_per_node,
            'mem_per_cpu': mem_per_cpu,
            'gpus_per_node': gpus_per_node
        }


def setLauncher(launcher, mpiTag = 'serial'):
    if mpiTag.lower() not in ["serial", "parallel"]:
        msg = '''The mpiTag passed should be either 'serial' or 'parallel'. The
        value passed is {}'''.format(mpiTag)
        raise ValueError(msg)

    if mpiTag.lower() == 'serial':
        return f'{launcher} -n 1'
