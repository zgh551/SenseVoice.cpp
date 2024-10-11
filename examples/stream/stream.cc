//
// Created by lovemefan on 2024/7/21.
//
#include "common.h"
#include "sense-voice.h"
#include <cmath>
#include <cstdint>
#include <thread>

#include "common-alsa.h"

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif


// command-line parameters
struct sense_voice_params {
    int32_t n_threads     = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors  = 1;
    int32_t offset_t_ms   = 0;
    int32_t duration_ms   = 0;
    int32_t max_context   = -1;
    int32_t best_of       = sense_voice_full_default_params(SENSE_VOICE_SAMPLING_GREEDY).greedy.best_of;
    int32_t beam_size     = sense_voice_full_default_params(SENSE_VOICE_SAMPLING_BEAM_SEARCH).beam_search.beam_size;
    // int32_t audio_ctx     = 0;
    int32_t step_ms       = 3000;
    int32_t length_ms     = 10000;
    int32_t keep_ms       = 200;
    int32_t capture_id    = -1;
    int32_t max_tokens    = 32;

    float vad_thold    = 0.5f;
    float freq_thold   = 100.0f;

    bool debug_mode      = false;
    bool print_progress  = false;
    bool no_timestamps   = false;
    bool save_audio      = false; // save audio to wav file
    bool use_gpu         = true;
    bool flash_attn      = false;

    std::string language  = "auto";
    std::string model     = "models/sense-voice-small-q4_0.gguf";
    std::string fname_out;
};

static int sense_voice_has_coreml(void) {
#ifdef SENSE_VOICE_USE_COREML
    return 1;
#else
    return 0;
#endif
}

static int sense_voice_has_openvino(void) {
#ifdef SENSE_VOICE_USE_OPENVINO
    return 1;
#else
    return 0;
#endif
}

const char * sense_voice_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "       + std::to_string(ggml_cpu_has_avx())       + " | ";
    s += "AVX2 = "      + std::to_string(ggml_cpu_has_avx2())      + " | ";
    s += "AVX512 = "    + std::to_string(ggml_cpu_has_avx512())    + " | ";
    s += "FMA = "       + std::to_string(ggml_cpu_has_fma())       + " | ";
    s += "NEON = "      + std::to_string(ggml_cpu_has_neon())      + " | ";
    s += "ARM_FMA = "   + std::to_string(ggml_cpu_has_arm_fma())   + " | ";
    s += "METAL = "     + std::to_string(ggml_cpu_has_metal())     + " | ";
    s += "F16C = "      + std::to_string(ggml_cpu_has_f16c())      + " | ";
    s += "FP16_VA = "   + std::to_string(ggml_cpu_has_fp16_va())   + " | ";
    s += "WASM_SIMD = " + std::to_string(ggml_cpu_has_wasm_simd()) + " | ";
    s += "BLAS = "      + std::to_string(ggml_cpu_has_blas())      + " | ";
    s += "SSE3 = "      + std::to_string(ggml_cpu_has_sse3())      + " | ";
    s += "SSSE3 = "     + std::to_string(ggml_cpu_has_ssse3())     + " | ";
    s += "VSX = "       + std::to_string(ggml_cpu_has_vsx())       + " | ";
    s += "CUDA = "      + std::to_string(ggml_cpu_has_cuda())      + " | ";
    s += "COREML = "    + std::to_string(sense_voice_has_coreml()) + " | ";
    s += "OPENVINO = "  + std::to_string(sense_voice_has_openvino());

    return s.c_str();
}

static void sense_voice_print_usage(int /*argc*/, char ** argv, const sense_voice_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options] file0.wav file1.wav ...\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,        --help              [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,      --threads N         [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "  -p N,      --processors N      [%-7d] number of processors to use during computation\n", params.n_processors);
    fprintf(stderr, "             --step N            [%-7d] audio step size in milliseconds\n",                params.step_ms);
    fprintf(stderr, "             --length N          [%-7d] audio length in milliseconds\n",                   params.length_ms);
    fprintf(stderr, "             --keep N            [%-7d] audio to keep from previous step in ms\n",         params.keep_ms);
    fprintf(stderr, "  -c ID,     --capture ID        [%-7d] capture device ID\n",                              params.capture_id);
    fprintf(stderr, "  -mt N,     --max-tokens N      [%-7d] maximum number of tokens per audio chunk\n",       params.max_tokens);
    fprintf(stderr, "  -vth N,    --vad-thold N       [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    fprintf(stderr, "  -fth N,    --freq-thold N      [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
    fprintf(stderr, "  -ot N,     --offset-t N        [%-7d] time offset in milliseconds\n",                    params.offset_t_ms);
    fprintf(stderr, "  -d  N,     --duration N        [%-7d] duration of audio to process in milliseconds\n",   params.duration_ms);
    fprintf(stderr, "  -mc N,     --max-context N     [%-7d] maximum number of text context tokens to store\n", params.max_context);
    fprintf(stderr, "  -bo N,     --best-of N         [%-7d] number of best candidates to keep\n",              params.best_of);
    fprintf(stderr, "  -bs N,     --beam-size N       [%-7d] beam size for beam search\n",                      params.beam_size);
    fprintf(stderr, "  -debug,    --debug-mode        [%-7s] enable debug mode (eg. dump log_mel)\n",           params.debug_mode ? "true" : "false");
    fprintf(stderr, "  -pp,       --print-progress    [%-7s] print progress\n",                                 params.print_progress ? "true" : "false");
    fprintf(stderr, "  -nt,       --no-timestamps     [%-7s] do not print timestamps\n",                        params.no_timestamps ? "true" : "false");
    fprintf(stderr, "  -l LANG,   --language LANG     [%-7s] spoken language ('auto' for auto-detect), support [`zh`, `en`, `yue`, `ja`, `ko`\n", params.language.c_str());
    fprintf(stderr, "  -m FNAME,  --model FNAME       [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -f FNAME,  --file FNAME        [%-7s] text output file name\n",                            "");
    fprintf(stderr, "  -sa,        --save-audio       [%-7s] save the recorded audio to a file\n",              params.save_audio ? "true" : "false");
    fprintf(stderr, "  -ng,       --no-gpu            [%-7s] disable GPU\n",                                    params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,       --flash-attn        [%-7s] flash attention\n",                                params.flash_attn ? "true" : "false");
    fprintf(stderr, "\n");
}

struct sense_voice_print_user_data {
    const sense_voice_params * params;

    const std::vector<std::vector<float>> * pcmf32s;
    int progress_prev;
};

static char * sense_voice_param_turn_lowercase(char * in){
    int string_len = strlen(in);
    for (int i = 0; i < string_len; i++){
        *(in+i) = tolower((unsigned char)*(in+i));
    }
    return in;
}

static bool sense_voice_params_parse(int argc, char ** argv, sense_voice_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            sense_voice_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")         { params.n_threads       = std::stoi(argv[++i]); }
        else if (arg == "-p"    || arg == "--processors")      { params.n_processors    = std::stoi(argv[++i]); }
        else if (                  arg == "--step")            { params.step_ms         = std::stoi(argv[++i]); }
        else if (                  arg == "--length")          { params.length_ms       = std::stoi(argv[++i]); }
        else if (                  arg == "--keep")            { params.keep_ms         = std::stoi(argv[++i]); }
        else if (arg == "-vth"  || arg == "--vad-thold")       { params.vad_thold       = std::stof(argv[++i]); }
        else if (arg == "-fth"  || arg == "--freq-thold")      { params.freq_thold      = std::stof(argv[++i]); }
        else if (arg == "-c"    || arg == "--capture")         { params.capture_id      = std::stoi(argv[++i]); }
        else if (arg == "-ot"   || arg == "--offset-t")        { params.offset_t_ms     = std::stoi(argv[++i]); }
        else if (arg == "-d"    || arg == "--duration")        { params.duration_ms     = std::stoi(argv[++i]); }
        else if (arg == "-mc"   || arg == "--max-context")     { params.max_context     = std::stoi(argv[++i]); }
        else if (arg == "-bo"   || arg == "--best-of")         { params.best_of         = std::stoi(argv[++i]); }
        else if (arg == "-bs"   || arg == "--beam-size")       { params.beam_size       = std::stoi(argv[++i]); }
        else if (arg == "-debug"|| arg == "--debug-mode")      { params.debug_mode      = true; }
        else if (arg == "-pp"   || arg == "--print-progress")  { params.print_progress  = true; }
        else if (arg == "-nt"   || arg == "--no-timestamps")   { params.no_timestamps   = true; }
        else if (arg == "-l"    || arg == "--language")        { params.language        = sense_voice_param_turn_lowercase(argv[++i]); }
        else if (arg == "-m"    || arg == "--model")           { params.model           = argv[++i]; }
        else if (arg == "-f"    || arg == "--file")            { params.fname_out       = argv[++i]; }
        else if (arg == "-sa"   || arg == "--save-audio")      { params.save_audio      = true; }
        else if (arg == "-ng"   || arg == "--no-gpu")          { params.use_gpu         = false; }
        else if (arg == "-fa"   || arg == "--flash-attn")      { params.flash_attn      = true; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            sense_voice_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

static bool is_file_exist(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

/**
 * This the arbitrary data which will be passed to each callback.
 * Later on we can for example add operation or tensor name filter from the CLI arg, or a file descriptor to dump the tensor.
 */
struct callback_data {
    std::vector<uint8_t> data;
};

static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        printf("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2*n) {
                printf("                                      ..., \n");
                i2 = ne[2] - n;
            }
            printf("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2*n) {
                    printf("                                       ..., \n");
                    i1 = ne[1] - n;
                }
                printf("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2*n) {
                        printf("..., ");
                        i0 = ne[0] - n;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v = 0;
                    if (type == GGML_TYPE_F16) {
                        v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[i]);
                    } else if (type == GGML_TYPE_F32) {
                        v = *(float *) &data[i];
                    } else if (type == GGML_TYPE_I32) {
                        v = (float) *(int32_t *) &data[i];
                    } else if (type == GGML_TYPE_I16) {
                        v = (float) *(int16_t *) &data[i];
                    } else if (type == GGML_TYPE_I8) {
                        v = (float) *(int8_t *) &data[i];
                    } else {
                        printf("fatal error");
                    }
                    printf("%12.4f", v);
                    sum += v;
                    if (i0 < ne[0] - 1) printf(", ");
                }
                printf("],\n");
            }
            printf("                                      ],\n");
        }
        printf("                                     ]\n");
        printf("                                     sum = %f\n", sum);
    }
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */
static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    if (ask) {
        return true; // Always retrieve data
    }

    char src1_str[128] = {0};
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
    }

    printf("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__,
           t->name, ggml_type_name(t->type), ggml_op_desc(t),
           src0->name, ggml_ne_string(src0).c_str(),
           src1 ? src1_str : "",
           ggml_ne_string(t).c_str());


    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    if (!ggml_is_quantized(t->type)) {
        uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
        ggml_print_tensor(data, t->type, t->ne, t->nb, 3);
    }

    return true;
}

void sense_voice_free(struct sense_voice_context * ctx) {
    if (ctx) {
        ggml_free(ctx->model.ctx);

        ggml_backend_buffer_free(ctx->model.buffer);

        sense_voice_free_state(ctx->state);

        delete ctx->model.model->encoder;
        delete ctx->model.model;
        delete ctx;
    }
}

int main(int argc, char ** argv) {
    sense_voice_params params;

    if (!sense_voice_params_parse(argc, argv, params)) {
        sense_voice_print_usage(argc, argv, params);
        return 1;
    }

    params.keep_ms   = std::min(params.keep_ms,   params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    const int n_samples_step = (1e-3*params.step_ms  )*SENSE_VOICE_SAMPLE_RATE;
    const int n_samples_len  = (1e-3*params.length_ms)*SENSE_VOICE_SAMPLE_RATE;
    const int n_samples_keep = (1e-3*params.keep_ms  )*SENSE_VOICE_SAMPLE_RATE;
    const int n_samples_30s  = (1e-3*30000.0         )*SENSE_VOICE_SAMPLE_RATE;

    const bool use_vad = n_samples_step <= 0; // sliding window mode uses VAD

    const int n_new_line = !use_vad ? std::max(1, params.length_ms / params.step_ms - 1) : 1; // number of steps to print new line

    params.no_timestamps  = !use_vad;
    // params.no_context    |= use_vad;
    params.max_tokens     = 0;

    HandleCtrlC();

    // init audio
    audio_cap audio(params.length_ms);
    if (!audio.init(params.capture_id, SENSE_VOICE_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }

    audio.resume();

    if (params.language != "auto" && sense_voice_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        sense_voice_print_usage(argc, argv, params);
        exit(0);
    }

    // sense-voice init

    struct sense_voice_context_params cparams = sense_voice_context_default_params();

    callback_data cb_data;

    cparams.cb_eval = ggml_debug;
    cparams.cb_eval_user_data = &cb_data;

    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    struct sense_voice_context * ctx = sense_voice_small_init_from_file_with_params(params.model.c_str(), cparams);

    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize sense voice context\n");
        return 3;
    }

    std::vector<double> pcmf32    (n_samples_30s, 0.0f);
    std::vector<double> pcmf32_old;
    std::vector<double> pcmf32_new(n_samples_30s, 0.0f);

    ctx->language_id = sense_voice_lang_id(params.language.c_str());

    {
        // print system information
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads*params.n_processors, std::thread::hardware_concurrency(), sense_voice_print_system_info());

        // print some info about the processing
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: processing audio (%d samples, %.5f sec) , %d threads, %d processors, lang = %s...\n",
                __func__,  int(pcmf32.size()), float(pcmf32.size())/SENSE_VOICE_SAMPLE_RATE,
                params.n_threads, params.n_processors,
                params.language.c_str());
        ctx->state->duration = float(pcmf32.size())/SENSE_VOICE_SAMPLE_RATE;
        fprintf(stderr, "\n");
    }

    int n_iter = 0;

    // bool is_running = true;

    // std::ofstream fout;
    // if (params.fname_out.length() > 0) {
    //     fout.open(params.fname_out);
    //     if (!fout.is_open()) {
    //         fprintf(stderr, "%s: failed to open output file '%s'!\n", __func__, params.fname_out.c_str());
    //         return 1;
    //     }
    // }

    wav_writer wavWriter;
    // save wav file
    if (params.save_audio) {
        // Get current date/time for filename
        time_t now = time(0);
        char buffer[80];
        strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", localtime(&now));
        std::string filename = std::string(buffer) + ".wav";

        wavWriter.open(filename, SENSE_VOICE_SAMPLE_RATE, 16, 1);
    }

    printf("[Start speaking]\n");
    fflush(stdout);

    auto t_last  = std::chrono::high_resolution_clock::now();
    const auto t_start = t_last;

    while (getSignal()) {
        // process new audio
        if (!getSignal()) {
            break;
        }

        if (!use_vad) {
            while (getSignal()) {
                audio.get(params.step_ms, pcmf32_new);

                if ((int) pcmf32_new.size() > 2*n_samples_step) {
                    fprintf(stderr, "\n\n%s: WARNING: cannot process audio fast enough, dropping audio ...\n\n", __func__);
                    audio.clear();
                    continue;
                }

                if ((int) pcmf32_new.size() >= n_samples_step) {
                    audio.clear();
                    break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            const int n_samples_new = pcmf32_new.size();

            // take up to params.length_ms audio from previous iteration
            const int n_samples_take = std::min((int) pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));

            //printf("processing: take = %d, new = %d, old = %d\n", n_samples_take, n_samples_new, (int) pcmf32_old.size());

            pcmf32.resize(n_samples_new + n_samples_take);

            for (int i = 0; i < n_samples_take; i++) {
                pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
            }

            memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(), n_samples_new*sizeof(float));

            pcmf32_old = pcmf32;
        } else {
            const auto t_now  = std::chrono::high_resolution_clock::now();
            const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();
            // wait 2s for sample audio
            if (t_diff < 2000) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                continue;
            }
            // get 2s sample data
            audio.get(2000, pcmf32_new);

            if (::vad_simple(pcmf32_new, SENSE_VOICE_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, false)) {
                // printf("active vad\n");
                audio.get(params.length_ms, pcmf32);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                continue;
            }
            t_last = t_now;
        }
        if (params.save_audio) {
            wavWriter.write(pcmf32.data(), pcmf32.size());
        }
        // printf("run the inference\n");

        sense_voice_full_params wparams = sense_voice_full_default_params(SENSE_VOICE_SAMPLING_GREEDY);
        // run inference
        {
            wparams.strategy = (params.beam_size > 1 ) ? SENSE_VOICE_SAMPLING_BEAM_SEARCH : SENSE_VOICE_SAMPLING_GREEDY;

            wparams.print_progress   = params.print_progress;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.language         = params.language.c_str();
            wparams.n_threads        = params.n_threads;
            wparams.n_max_text_ctx   = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
            wparams.offset_ms        = params.offset_t_ms;
            wparams.duration_ms      = params.duration_ms;

            wparams.debug_mode       = params.debug_mode;

            wparams.greedy.best_of        = params.best_of;
            wparams.beam_search.beam_size = params.beam_size;

            wparams.no_timestamps    = params.no_timestamps;

            if (sense_voice_full_parallel(ctx, wparams, pcmf32, pcmf32.size(), params.n_processors) != 0) {
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                return 10;
            }

            // print result;
            {
                if (!use_vad) {
                    printf("\33[2K\r");

                    // print long empty line to clear the previous line
                    printf("%s", std::string(100, ' ').c_str());

                    printf("\33[2K\r");
                } else {
                    const int64_t t1 = (t_last - t_start).count()/1000000;
                    const int64_t t0 = std::max(0.0, t1 - pcmf32.size()*1000.0/SENSE_VOICE_SAMPLE_RATE);

                    printf("\n");
                    printf("### Transcription %d START | t0 = %d ms | t1 = %d ms\n", n_iter, (int) t0, (int) t1);
                    printf("\n");
                }

                // std::string result;

                const int n_segments = sense_voice_full_n_segments(ctx);
                for (int i = 0; i < n_segments; ++i) {
                    const char * text = sense_voice_full_get_segment_text(ctx, i);
                    printf("%s", text);
                    // fflush(stdout);
                }
                printf("\n");
                fflush(stdout);
                // if (params.fname_out.length() > 0) {
                //     fout << std::endl;
                // }

                if (use_vad) {
                    printf("\n");
                    printf("### Transcription %d END\n", n_iter);
                }
            }

            ++n_iter;

            if (!use_vad && (n_iter % n_new_line) == 0) {
                printf("\n");
                // keep part of the audio for next iteration to try to mitigate word boundary issues
                pcmf32_old = std::vector<double>(pcmf32.end() - n_samples_keep, pcmf32.end());
            }
            fflush(stdout);
        }
    }

    audio.pause();
    sense_voice_free(ctx);
    return 0;
}