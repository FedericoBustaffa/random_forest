#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <cstdio>
#include <string>

using milli = std::chrono::milliseconds;
using micro = std::chrono::microseconds;
using nano = std::chrono::nanoseconds;

template <typename Duration = std::chrono::seconds>
class Timer
{

public:
    using clock = std::chrono::high_resolution_clock;

    using duration = std::chrono::duration<double, typename Duration::period>;

    Timer() = default;

    inline void start() { m_Start = clock::now(); }

    inline double stop()
    {
        auto elapsed = duration(clock::now() - m_Start);
        return elapsed.count();
    }

    inline double stop(const std::string& label)
    {
        double elapsed = stop();
        std::printf("%s time: %.2f %s\n", label.c_str(), elapsed, suffix());

        return elapsed;
    }

private:
    static constexpr const char* suffix()
    {
        if constexpr (std::is_same_v<Duration, milli>)
            return "ms";
        else if constexpr (std::is_same_v<Duration, micro>)
            return "us";
        else if constexpr (std::is_same_v<Duration, nano>)
            return "ns";
        else
            return "s";
    }

private:
    std::chrono::time_point<clock> m_Start;
};

#endif
