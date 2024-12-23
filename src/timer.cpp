#include "timer.h"



Timer::Timer(int64_t ms) : duration(ms), start_time(std::chrono::high_resolution_clock::now()) {};

int64_t Timer::time_remaining() {

    auto now = std::chrono::high_resolution_clock::now();
    auto rem = duration - std::chrono::duration_cast<std::chrono::milliseconds>(now - this->start_time).count();

    if (rem > 0) {
        return rem;
    }
    return 0;
}

int64_t Timer::time_elapsed() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - this->start_time).count();
}