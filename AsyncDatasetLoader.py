import torch
from queue import Queue
from threading import Thread, Event


class AsyncCudaDataLoader:
    """ 异步预先将数据从 CPU 加载到 GPU 中 """

    def __init__(self, loader, device="cpu", queue_size=2, name=None):
        self.device = device
        self.queue_size = queue_size
        self.loader = loader
        self.stop = Event()

        if "cuda" in str(device):
            self.load_stream = torch.cuda.Stream(device=device)
        else:
            self.load_stream = None
        self.queue = Queue(maxsize=self.queue_size)

        self.idx = 0
        self.worker = Thread(target=self.load_loop, daemon=True, name=name)
        self.worker.start()

    def __del__(self):
        self.stop.set()
        if not self.queue.empty():
            self.queue.get()
        self.worker.join()
        self.queue.queue.clear()

    def terminate(self):
        self.__del__()

    def load_instance(self, sample):
        """ 将 batch 数据从 CPU 加载到 GPU 中 """
        if torch.is_tensor(sample):
            if self.load_stream:
                with torch.cuda.stream(self.load_stream):
                    return sample.to(self.device, non_blocking=True)
            else:
                return sample.to(self.device, non_blocking=True)
        elif isinstance(sample, dict):
            return {k: self.load_instance(v) for k, v in sample.items()}
        elif isinstance(sample, list):
            return [self.load_instance(s) for s in sample]
        elif isinstance(sample, tuple):
            return (self.load_instance(s) for s in sample)
        return sample

    def load_loop(self):
        """ 不断的将 cuda 数据加载到队列里 """
        # The loop that will load into the queue in the background
        while not self.stop.is_set():
            for i, sample in enumerate(self.loader):
                if self.stop.is_set():
                    break
                tmp = self.load_instance(sample)
                self.queue.put(tmp)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        # 加载线程意外退出了
        if not self.worker.is_alive() and self.queue.empty():
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        # 一个 epoch 加载完了
        elif self.idx >= len(self.loader):
            self.idx = 0
            raise StopIteration
        # 下一个 batch
        else:
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out

    def next(self):
        return self.__next__()

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset
