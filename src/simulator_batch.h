#include "simulator.h"
#include <queue>
#include <pthread.h>


class SimulatorBatch {
public:

    SimulatorBatch(int num_processes=4);
    ~SimulatorBatch();

    int get_queue_size();
    void add(Simulator& game);
    void exit_thread();

    friend void* simulator_worker(void* arg);

private:

    bool thread_exit;

    std::queue<Simulator*> input_queue;
    std::vector<pthread_t> worker_threads;
    pthread_mutex_t lock;
    pthread_cond_t input_added;
    pthread_cond_t finished_game;
};

