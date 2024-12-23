#include "simulator.h"
#include <queue>
#include <pthread.h>


class SimulatorBatch {
public:

    SimulatorBatch(int num_processes=4);
    ~SimulatorBatch();

    int total_remaining();
    void add(Simulator& game);

    friend void* simulator_worker(void* arg);

private:

    bool thread_exit;
    int processes_remaining = 0;
    std::queue<Simulator*> input_queue;
    std::vector<pthread_t> worker_threads;
    pthread_mutex_t lock;
    pthread_cond_t input_added;
    pthread_cond_t finished_game;
};

