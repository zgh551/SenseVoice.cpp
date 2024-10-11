#include "common-alsa.h"
#include <sys/epoll.h>

#include <atomic>
#include <mutex>
#include <condition_variable>
#include "common.h"

audio_cap::audio_cap(int len_ms) {
    m_len_ms = len_ms;

    m_running = false;
}

audio_cap::~audio_cap() {
    m_thread.join();
    if (m_pcm) {
        pcm_close(m_pcm);
    }
}

bool audio_cap::init(int capture_id, int sample_rate) {
    memset(&m_config, 0, sizeof(m_config));
    m_config.channels          = 4;
    m_config.rate              = sample_rate;
    m_config.period_size       = 1024;
    m_config.period_count      = 4;
    m_config.format            = PCM_FORMAT_S16_LE;
    m_config.start_threshold   = 0;
    m_config.stop_threshold    = 0;
    m_config.silence_threshold = 0;

    m_pcm = pcm_open(capture_id, 0, PCM_IN, &m_config);

    if (!m_pcm || !pcm_is_ready(m_pcm)) {
        fprintf(stderr, "Unable to open PCM device (%s)\n",
                pcm_get_error(m_pcm));
        return 0;
    }

    m_frames_read = pcm_get_buffer_size(m_pcm);
    m_byte_read   = pcm_frames_to_bytes(m_pcm, m_frames_read);
    fprintf(stderr, "read %u frames; read %u bytes\n", m_frames_read, m_byte_read);
    m_buffer = malloc(m_byte_read);
    if (!m_buffer) {
        fprintf(stderr, "Unable to allocate %u bytes\n", m_byte_read);
        pcm_close(m_pcm);
        return 0;
    }
    m_sample_rate = sample_rate;
    m_audio.resize((m_sample_rate * m_len_ms) / 1000);
    m_period_audio.resize(m_frames_read);

    m_thread = std::thread(&audio_cap::audio_thread_func, this);

    return true;
}

void audio_cap::audio_thread_func() {
    int epfd = epoll_create1(0);

    struct epoll_event ev, events[1];

    ev.events = EPOLLIN; // 监听可读事件
    ev.data.fd = pcm_get_poll_fd(m_pcm);

    epoll_ctl(epfd, EPOLL_CTL_ADD, pcm_get_poll_fd(m_pcm), &ev);

    while (getSignal()) {
        int nfds = epoll_wait(epfd, events, 1, -1); // 阻塞等待事件

        if (nfds > 0) {
            if (events[0].events & EPOLLIN) {
                int st = pcm_read(m_pcm, m_buffer, m_byte_read);
                if (0 == st) {
                    for (int i = 0; i < m_frames_read; i++) {
                        m_period_audio[i] = *(static_cast<int16_t *>(m_buffer) + 4 * i) / 1.0;
                    }
                    callback(reinterpret_cast<uint8_t*>(m_period_audio.data()), m_frames_read * 8);
                }
            }
        }
    }
    close(epfd);
}

// callback to be called by alsa
void audio_cap::callback(uint8_t * stream, size_t len) {
    if (!m_running) {
        return;
    }

    size_t n_samples = len / sizeof(double);

    if (n_samples > m_audio.size()) {
        n_samples = m_audio.size();

        stream += (len - (n_samples * sizeof(double)));
    }

    //fprintf(stderr, "%s: %zu samples, pos %zu, len %zu\n", __func__, n_samples, m_audio_pos, m_audio_len);

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_audio_pos + n_samples > m_audio.size()) {
            const size_t n0 = m_audio.size() - m_audio_pos;

            memcpy(&m_audio[m_audio_pos], stream, n0 * sizeof(double));
            memcpy(&m_audio[0], stream + n0 * sizeof(double), (n_samples - n0) * sizeof(double));

            m_audio_pos = (m_audio_pos + n_samples) % m_audio.size();
            m_audio_len = m_audio.size();
        } else {
            memcpy(&m_audio[m_audio_pos], stream, n_samples * sizeof(double));

            m_audio_pos = (m_audio_pos + n_samples) % m_audio.size();
            m_audio_len = std::min(m_audio_len + n_samples, m_audio.size());
        }
    }
}

bool audio_cap::resume() {
    printf("resume\n");
    if (!m_pcm) {
        fprintf(stderr, "%s: no audio device to resume!\n", __func__);
        return false;
    }

    if (m_running) {
        fprintf(stderr, "%s: already running!\n", __func__);
        return false;
    }

    int ret = pcm_start(m_pcm);
    if (0 == ret) {
        m_running = true;

        return true;
    } else {
        return false;
    }
}

bool audio_cap::pause() {
    printf("pause\n");
    if (!m_pcm) {
        fprintf(stderr, "%s: no audio device to pause!\n", __func__);
        return false;
    }

    if (!m_running) {
        fprintf(stderr, "%s: already paused!\n", __func__);
        return false;
    }

    int ret = pcm_stop(m_pcm);
    if (0 == ret) {
        m_running = false;

        return true;
    } else {
        return false;
    }
    m_thread.join();
}

bool audio_cap::clear() {
    if (!m_pcm) {
        fprintf(stderr, "%s: no audio device to clear!\n", __func__);
        return false;
    }

    if (!m_running) {
        fprintf(stderr, "%s: not running!\n", __func__);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        m_audio_pos = 0;
        m_audio_len = 0;
    }

    return true;
}

#if 1
void audio_cap::get(int ms, std::vector<double> & result) {
    if (!m_running) {
        fprintf(stderr, "%s: not running!\n", __func__);
        return;
    }

    result.clear();

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (ms <= 0) {
            ms = m_len_ms;
        }

        size_t n_samples = (m_sample_rate * ms) / 1000;
        if (n_samples > m_audio_len) {
            n_samples = m_audio_len;
        }

        result.resize(n_samples);

        int s0 = m_audio_pos - n_samples;
        if (s0 < 0) {
            s0 += m_audio.size();
        }

        if (s0 + n_samples > m_audio.size()) {
            const size_t n0 = m_audio.size() - s0;

            memcpy(result.data(), &m_audio[s0], n0 * sizeof(double));
            memcpy(&result[n0], &m_audio[0], (n_samples - n0) * sizeof(double));
        } else {
            memcpy(result.data(), &m_audio[s0], n_samples * sizeof(double));
        }
    }
}

#else
void audio_cap::get(int ms, std::vector<double> &result) {
    // printf("[Start get]\n");
    if (!m_pcm) {
        fprintf(stderr, "%s: no audio device to get audio from!\n", __func__);
        return;
    }

    if (!m_running) {
        fprintf(stderr, "%s: not running!\n", __func__);
        return;
    }

    result.clear();
    uint32_t frames_read = 0, total_frames_read = 0, byte_size=0;
    bool capturing = true;
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (ms <= 0) {
            ms = m_len_ms;
        }
        size_t n_samples = (m_sample_rate * ms) / 1000;
        // if (n_samples > m_audio_len) {
        //     n_samples = m_audio_len;
        // }
        result.resize(n_samples);
        // byte_size = pcm_frames_to_bytes(m_pcm, pcm_get_buffer_size(m_pcm));
        // frames_read = pcm_bytes_to_frames(m_pcm, byte_size);
        while (capturing) {
            // if (pcm_wait(m_pcm, -1) < 0 && errno != ETIMEDOUT) {
            //     fprintf(stderr, "pcm_wait failed: %s\n", strerror(errno));
            //     break;
            // }
            pcm_read(m_pcm, m_buffer, m_byte_read);
            for (int i = 0; i < m_frames_read; i++) {
                if (total_frames_read + i < n_samples) {
                    result[total_frames_read + i] = *(static_cast<int16_t *>(m_buffer) + 4 * i) / 1.0;
                } else {
                    break;
                }
            }
            // fprintf(stderr, "%s: audio device to get audio from! size=%d; total=%d\n", __func__, byte_size, total_frames_read);
            total_frames_read += m_frames_read;
            if ((total_frames_read / m_sample_rate)*1000 >= ms) {
                capturing = false;
            }
        }
    }
}
#endif
