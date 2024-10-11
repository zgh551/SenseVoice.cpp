//
// Created by lovemefan on 2024/7/21.
//
//
// logging
//
#include <dirent.h>
#include <pthread.h>
#include <signal.h>
#include <atomic>
#include "common.h"
#include <stdarg.h>
#ifdef __GNUC__
#ifdef __MINGW32__
#define SENSEVOICE_ATTRIBUTE_FORMAT(...) \
  __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define SENSEVOICE_ATTRIBUTE_FORMAT(...) \
  __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define SENSEVOICE_ATTRIBUTE_FORMAT(...)
#endif

void sense_voice_log_callback_default(ggml_log_level level,
                                             const char *text, void *user_data) {
    (void)level;
    (void)user_data;
    fputs(text, stderr);
    fflush(stderr);
}

GGML_ATTRIBUTE_FORMAT(2, 3)
void sense_voice_log_internal(ggml_log_level level, const char *format,
                                     ...) {
    va_list args;
    va_start(args, format);
    char buffer[1024];
    int len = vsnprintf(buffer, 1024, format, args);
    if (len < 1024) {
        g_state.log_callback(level, buffer, g_state.log_callback_user_data);
    } else {
        char *buffer2 = new char[len + 1];
        vsnprintf(buffer2, len + 1, format, args);
        buffer2[len] = 0;
        g_state.log_callback(level, buffer2, g_state.log_callback_user_data);
        delete[] buffer2;
    }
    va_end(args);
}


struct sense_voice_full_params sense_voice_full_default_params(enum sense_voice_decoding_strategy strategy) {
    struct sense_voice_full_params result = {
            /*.strategy          =*/ strategy,

            /*.n_threads         =*/ std::min(4, (int32_t) std::thread::hardware_concurrency()),
            /* language          =*/ "auto",
            /*.n_max_text_ctx    =*/ 16384,
            /*.offset_ms         =*/ 0,
            /*.duration_ms       =*/ 0,

            /*.no_context        =*/ true,
            /*.no_timestamps     =*/ false,
            /*.print_progress    =*/ true,
            /*.print_timestamps  =*/ true,


            /*.debug_mode        =*/ false,
            /* audio_ctx         =*/ 0,

            /*.greedy            =*/ {
                    /*.best_of   =*/ -1,
            },

            /*.beam_search      =*/ {
                    /*.beam_size =*/ -1,
            },

            /*.progress_callback           =*/ nullptr,
            /*.progress_callback_user_data =*/ nullptr,

    };

    switch (strategy) {
        case SENSE_VOICE_SAMPLING_GREEDY:
        {
            result.greedy = {
                    /*.best_of   =*/ 5,
            };
        } break;
        case SENSE_VOICE_SAMPLING_BEAM_SEARCH:
        {
            result.beam_search = {
                    /*.beam_size =*/ 5
            };
        } break;
    }

    return result;
}

void high_pass_filter(std::vector<double> & data, float cutoff, float sample_rate) {
    const float rc = 1.0f / (2.0f * M_PI * cutoff);
    const float dt = 1.0f / sample_rate;
    const float alpha = dt / (rc + dt);

    float y = data[0];

    for (size_t i = 1; i < data.size(); i++) {
        y = alpha * (y + data[i] - data[i - 1]);
        data[i] = y;
    }
}

bool vad_simple(std::vector<double> & pcmf32, int sample_rate, int last_ms, float vad_thold, float freq_thold, bool verbose) {
    const int n_samples      = pcmf32.size();
    const int n_samples_last = (sample_rate * last_ms) / 1000;

    if (n_samples_last >= n_samples) {
        // not enough samples - assume no speech
        return false;
    }

    if (freq_thold > 0.0f) {
        high_pass_filter(pcmf32, freq_thold, sample_rate);
    }

    float energy_all  = 0.0f;
    float energy_last = 0.0f;

    for (int i = 0; i < n_samples; i++) {
        energy_all += fabsf(pcmf32[i]);

        if (i >= n_samples - n_samples_last) {
            energy_last += fabsf(pcmf32[i]);
        }
    }

    energy_all  /= n_samples;
    energy_last /= n_samples_last;

    if (verbose) {
        fprintf(stderr, "%s: energy_all: %f, energy_last: %f, vad_thold: %f, freq_thold: %f\n", __func__, energy_all, energy_last, vad_thold, freq_thold);
    }

    if (energy_last > vad_thold*energy_all) {
        return false;
    }

    return true;
}

std::atomic<bool> run_flag(true);

bool getSignal(void) { return run_flag.load(); }

void signal_hdl(int32_t sig) { run_flag.store(false); }

void* IntSingleFn(void* arg) {
    int err   = 0;
    int signo = 0;

    sigset_t waitset, oldset;

    sigemptyset(&waitset);
    sigaddset(&waitset, SIGINT);
    sigaddset(&waitset, SIGTSTP);  // ctrl-z

    if ((err = pthread_sigmask(SIG_BLOCK, &waitset, &oldset)) != 0) {
        fprintf(stderr, "pthread_sigmask error: %d\n", err);
    }
    while (run_flag.load()) {
        printf("sub thread(%lu) sigwait for signal...\n", pthread_self());
        err = sigwait(&waitset, &signo);
        if (err != 0) {
            perror("sigwait error");
            exit(err);
        }
        signal_hdl(signo);
    }
    return nullptr;
}

void HandleCtrlC(void) {
    pthread_t iThread;
    sigset_t mask, oldmask;
    int err;
    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGTSTP);
    if ((err = pthread_sigmask(SIG_BLOCK, &mask, &oldmask)) != 0) {
        fprintf(stderr, "pthread_sigmask error: %d\n", err);
    }
    if ((err = pthread_create(&iThread, NULL, IntSingleFn, NULL)) != 0) {
        fprintf(stderr, "pthread_sigmask error: %d\n", err);
    }
}
