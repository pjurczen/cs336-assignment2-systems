from threading import Thread, Event

from psutil import Process, NoSuchProcess


class PeekMemorySampler:
    stop: Event
    thread: Thread
    peak: int

    def __enter__(self) -> "PeekMemorySampler":
        self.stop = Event()
        self.peak = 0

        def sampler():
            found_workers: bool = False
            main_process: Process = Process()
            worker_processes: list[Process] = []
            while not self.stop.is_set():
                if not found_workers and not worker_processes:
                    worker_processes = main_process.children(recursive=True)
                    if worker_processes:
                        found_workers = True
                procs = [main_process] + worker_processes
                current: int = 0
                for p in procs:
                    try:
                        current += p.memory_info().rss
                    except NoSuchProcess:
                        worker_processes.remove(p)
                self.peak = max(self.peak, current)
                self.stop.wait(0.1)

        self.thread = Thread(target=sampler, daemon=True)
        self.thread.start()  # sampler begins running NOW

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop.set()  # tell it to finish
        self.thread.join()  # wait until it has

    def peak_mib(self) -> int:
        return self.peak / 1024 ** 2
