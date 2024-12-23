#ifndef TIMER_H
#define TIMER_H


#include <iostream>
#include <chrono>
#include <thread>

class Timer {
public:
    Timer(int64_t ms);
    int64_t time_remaining();
    int64_t time_elapsed();
private:
    int64_t duration; // Total duration of the timer
    std::chrono::high_resolution_clock::time_point start_time; // Timer start time
};

#endif 