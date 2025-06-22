//ovde treba da napravam tajmeridx

#pragma once

#include <chrono>

class Timer
{
	private:
		typedef std::chrono::time_point<
			std::chrono::high_resolution_clock,
			std::chrono::nanoseconds
		> timestamp;

		timestamp start_timestamp;
		timestamp stop_timestamp;

	public:
		Timer() { };

		void start()
		{
			this->start_timestamp = std::chrono::high_resolution_clock::now();
		}
		uint64_t stop()
		{
			this->stop_timestamp = std::chrono::high_resolution_clock::now();
			return std::chrono::duration<uint64_t, std::nano>(
				stop_timestamp - start_timestamp
			).count();
		}
};
