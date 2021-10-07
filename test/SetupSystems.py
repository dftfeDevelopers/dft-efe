import re
from reframe.core.runtime import runtime
#import mysettings as config
#system_partition_list = ["greatlakes:standard", "greatlakes:gpu", "greatlakes:login", 
#                         "generic:default"]

def getSystemPartitionList():
    partitions = runtime().system.partitions
    return [p.fullname for p in partitions]

def getValidSystems(key):
    
    system_partition_list = getSystemPartitionList()
    returnVal = []
    if key.lower() not in ["cpu", "gpu", "both"]:
        msg = '''The tag passed should be one of: 'cpu', 'gpu', or
        'both'. Tag passed is {}'''.format(key)
        raise ValueError(msg)
    
    else:
        if key.lower() == "cpu":
            sys_part_string = "\n".join(system_partition_list)
            for match in re.finditer('^((?!gpu).)*$', sys_part_string, flags=re.M):
                returnVal.append(match.group(0))

        elif key.lower() == "gpu":
            sys_part_string = "\n".join(system_partition_list)
            for match in re.finditer(r'.*gpu.*', sys_part_string, flags=re.M):
                returnVal.append(match.group(0))
        
        else:
            returnVal = system_partition_list

        return returnVal

def setResources(archTag = 'both', time_limit = "00:02:00", num_nodes = 1, num_tasks_per_node = 1, mem_per_cpu =
                 '2gb', gpus_per_node = 1):
    if archTag.lower() not in ["cpu", "gpu", "both"]:
        msg = '''The tag passed should be one of: 'cpu', 'gpu', or
        'both'. Tag passed is {}'''.format(archTag)
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

    if archTag.lower() == 'both':
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

    return returnVal
