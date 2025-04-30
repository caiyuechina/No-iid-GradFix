import os
import psutil
import multiprocessing as mp

try:
    from .myaffinity import set_process_affinity_mask
except:
    set_process_affinity_mask = None


def setSelf(priority=None, cpuList=None):
    p = psutil.Process(os.getpid())
    if priority:
        p.nice(priority)
    if cpuList and set_process_affinity_mask is not None:
        set_process_affinity_mask(os.getpid(), cpuAffinityBuilder(cpuList))


def workerInit(priority=None, cpuID=None):
    p = psutil.Process(os.getpid())
    if priority:
        p.nice(priority)
    if cpuID and set_process_affinity_mask is not None:
        set_process_affinity_mask(os.getpid(), cpuID)


def initPool(poolSize, priorityList=None, cpuList=None):
    if cpuList:
        assert len(cpuList) == poolSize
        cpuList = list(map(cpuAffinityBuilder, cpuList))
    else:
        cpuList = [None for _ in range(0, poolSize)]
    if priorityList:
        if isinstance(priorityList, int):
            priorityList = [priorityList for _ in range(0, poolSize)]
        assert len(priorityList) == poolSize
    else:
        priorityList = [None for _ in range(0, poolSize)]
    pool = mp.Pool(poolSize)
    initJobs = []
    for i in range(0, poolSize):
        job = pool.apply_async(workerInit, (priorityList[i], cpuList[i]))
        initJobs.append(job)
    for job in initJobs:
        job.wait()
    return pool


def cpuAffinityBuilder(cpuSelectList):
    import psutil
    assert isinstance(cpuSelectList, list)
    cpucode = ['0' for _ in range(0, psutil.cpu_count())]
    for sel in cpuSelectList:
        assert isinstance(sel, int)
        cpucode[sel] = '1'
    cpucode.reverse()
    codeStr = "".join(cpucode)
    return int(codeStr, 2)
