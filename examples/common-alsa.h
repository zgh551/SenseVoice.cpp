#pragma once

#include <limits.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tinyalsa/asoundlib.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>
#include <thread>

//
// Tinyalsa Audio capture
//

class audio_cap {
public:
    audio_cap(int len_ms);
    ~audio_cap();

    bool init(int capture_id, int sample_rate);

    // start capturing audio via the provided SDL callback
    // keep last len_ms seconds of audio in a circular buffer
    bool resume();
    bool pause();
    bool clear();

    // get audio data from the circular buffer
    void get(int ms, std::vector<double>& audio);

private:
    void audio_thread_func();

    // callback to be called by SDL
    void callback(uint8_t* stream, size_t len);

private:
    // SDL_AudioDeviceID m_dev_id_in = 0;
    struct pcm_config m_config;
    struct pcm* m_pcm;
    void* m_buffer;

    int m_len_ms      = 0;
    int m_sample_rate = 0;

    std::atomic_bool m_running;
    std::mutex m_mutex;

    uint32_t m_frames_read = 0;
    uint32_t m_byte_read = 0;

    std::vector<double> m_audio;
    std::vector<double> m_period_audio;
    size_t m_audio_pos = 0;
    size_t m_audio_len = 0;

    std::thread m_thread;
};

