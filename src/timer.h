#ifndef TIMER_H
#define TIMER_H


#include <iostream>
#include <chrono>
#include <thread>

class Timer {
public:
    Timer(int ms);
    int time_remaining();
    int time_elapsed();
private:
    int duration; // Total duration of the timer
    std::chrono::high_resolution_clock::time_point start_time; // Timer start time
};

#endif 