import os
import sys
import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def reduce_gradidents(params, world_size):
    if world_size == 1:
        return
    all_grads_list = []
    for param in params:
        if param.grad is not None:
            all_grads_list.append(param.grad.view(-1))
    all_grads = torch.cat(all_grads_list)
    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
    offset = 0
    for param in params:
        if param.grad is not None:
            param.grad.data.copy_(
                all_grads[offset : offset + param.numel()].view_as(param.grad.data) / world_size
            )
            offset += param.numel()


def test_nccl(local_rank):
    # manual init nccl
    x = torch.rand(4, device=f'cuda:{local_rank}')
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x.mean().item()
    dist.barrier()


def torchrun_setup(backend, local_rank):
    dist.init_process_group(
        backend, timeout=datetime.timedelta(seconds=60 * 30))
    test_nccl(local_rank)


def setup(backend, rank, world_size, port):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(
        backend, rank=rank, world_size=world_size,
        timeout=datetime.timedelta(seconds=60 * 30))

    test_nccl(rank)


def mp_start(run):
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size == 1:
        run(local_rank=0, world_size=world_size)
    else:
        # mp.set_start_method('spawn')
        children = []
        for i in range(world_size):
            subproc = mp.Process(target=run, args=(i, world_size))
            children.append(subproc)
            subproc.start()

        for i in range(world_size):
            children[i].join()


def fprint(msg):
    sys.stdout.flush()
    sys.stdout.write(msg + os.linesep)
    sys.stdout.flush()


class SummaryWriter:

    def __init__(self, log_dir, debug=False):
        self.log_dir = log_dir
        self.debug = debug
        if not debug:
            from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
            self.writer = _SummaryWriter(log_dir)
        else:
            self.writer = None

    def __getattr__(self, item):
        if self.debug:
            return lambda *args, **kwargs: None
        return getattr(self.writer, item)


class SummaryWriterProxy:

    def __init__(self, log_queue):
        self.log_queue = log_queue

    def add_scalar(self, *args, **kwargs):
        self.log_queue.put(('scalar', args, kwargs))

    def add_text(self, *args, **kwargs):
        self.log_queue.put(('text', args, kwargs))

    def close(self):
        self.log_queue.put(('_close', None, None))


def tb_worker_fn(queue, log_dir, debug):
    writer = SummaryWriter(log_dir, debug)
    try:
        while True:
            log_type, args, kwargs = queue.get()
            if log_type == 'scalar':
                writer.add_scalar(*args, **kwargs)
            elif log_type == 'text':
                writer.add_text(*args, **kwargs)
            elif log_type == 'close':
                break
    except KeyboardInterrupt:
        pass
    finally:
        writer.flush()
        writer.close()


class MpSummaryWriterManager:

    def __init__(self, log_dir, debug=False):
        self.log_dir = log_dir
        self.debug = debug

        self.queue = mp.SimpleQueue()
        self.worker = mp.Process(target=tb_worker_fn, args=(self.queue, log_dir, debug), daemon=True)
        self.worker.start()

    def get_queue(self):
        return self.queue
    
    def close(self):
        self.queue.put(('close', None, None))
        self.worker.join()