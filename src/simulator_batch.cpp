#include "simulator_batch.h"



void* simulator_worker(void* arg) {
    SimulatorBatch* batch = static_cast<SimulatorBatch*>(arg);

    while (true) {

        pthread_mutex_lock(&batch->lock);

        while (batch->input_queue.empty() && batch->thread_exit == false) {
            pthread_cond_wait(&batch->input_added, &batch->lock);
        }

        if (batch->thread_exit == true) {  
            pthread_mutex_unlock(&batch->lock);
            break;
        }

        Simulator* game = batch->input_queue.front();    
        batch->input_queue.pop();
        pthread_mutex_unlock(&batch->lock);
        game->run();

        pthread_mutex_lock(&batch->lock);
        pthread_cond_broadcast(&batch->finished_game);
        pthread_mutex_unlock(&batch->lock);
    }

    return nullptr;
}


SimulatorBatch::SimulatorBatch(int num_processes) {

    this->thread_exit = false;

    pthread_mutex_init(&this->lock, nullptr);
    pthread_cond_init(&this->input_added, nullptr);
    pthread_cond_init(&this->finished_game, nullptr);

    for (int i = 0; i < num_processes; i++) {
        pthread_t thread;
        int result = pthread_create(&thread, NULL, &simulator_worker, this);
        if (result != 0) {
            std::cerr << "Error: SimulationBatch pthread_create failed" << std::endl;
            exit(1);
        }
        this->worker_threads.push_back(thread);
    }

}



int SimulatorBatch::get_queue_size() {

    int size = 0;
    pthread_mutex_lock(&this->lock);
    size = this->input_queue.size();
    pthread_mutex_unlock(&this->lock);
    return size;
}



void SimulatorBatch::add(Simulator& game) {

   
    pthread_mutex_lock(&this->lock);
    this->input_queue.push(&game);
    pthread_cond_signal(&this->input_added);
    pthread_mutex_unlock(&this->lock);

}



SimulatorBatch::~SimulatorBatch() {
    this->exit_thread();
}


void SimulatorBatch::exit_thread() {
    if (this->thread_exit == false) {
        this->thread_exit = true;
        pthread_cond_broadcast(&this->input_added);
        
        for ( auto thread : this->worker_threads ) {
            pthread_join(thread, nullptr);
        }

        pthread_mutex_destroy(&this->lock);
        pthread_cond_destroy(&this->input_added);
        pthread_cond_destroy(&this->finished_game);
    }
}