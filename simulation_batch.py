from game_simulator import GameSimulator
import threading
import queue

def game_worker(sim):
    while True:
        game = None

        with sim.lock:
            while sim.queue.empty() and not sim.thread_exit:
                sim.added.wait()
            if sim.thread_exit:
                return
            game = sim.queue.get()

        game.run()

        with sim.lock:
            sim.finished.notify_all()


class SimulationBatch:

    def __init__(self, num_threads=4):
        self.thread_exit = False
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.added = threading.Condition(self.lock)
        self.finished = threading.Condition(self.lock)

        self.threads = [
            threading.Thread(target=game_worker, args=(self,)) for i in range(num_threads)
        ]

        for thread in self.threads:
            thread.start()

    def add_game(self, white_bot, black_bot, move_time=300, move_limit=200):
        with self.lock:
            game = GameSimulator(white_bot, black_bot, move_time=move_time, move_limit=move_limit)
            self.queue.put(game)
            self.added.notify()
            return game

    def get_queue_size(self):
        with self.lock:
            return self.queue.qsize()

    def wait_finish(self):
        with self.lock:
            while not self.queue.empty():
                self.finished.wait()

    def exit(self):
        with self.lock:
            self.thread_exit = True
            self.added.notify_all()

        for thread in self.threads:
            thread.join()
