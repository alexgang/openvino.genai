#include "chatglm.h"
#include <fstream>
#include <iomanip>
#include <iostream>

#ifdef _WIN32
#include <codecvt>
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

enum InferenceMode {
    INFERENCE_MODE_CHAT,
    INFERENCE_MODE_GENERATE,
};

static inline InferenceMode to_inference_mode(const std::string &s) {
    static std::unordered_map<std::string, InferenceMode> m{{"chat", INFERENCE_MODE_CHAT},
                                                            {"generate", INFERENCE_MODE_GENERATE}};
    return m.at(s);
}

//struct Args {
//    std::string model_path = "..\\..\\..\\..\\..\\..\\ChatGLM3_C++_demo\\model\\GPTQ_INT4-FP16\\";
//    InferenceMode mode = INFERENCE_MODE_CHAT;
//    bool sync = false;
//    std::string prompt = "你好";
//    std::string system = "";
//    int max_length = 2048;
//    int max_new_tokens = -1;
//    int max_context_length = 512;
//    bool interactive = false;
//    int top_k = 0;
//    float top_p = 0.7;
//    float temp = 0.95;
//    float repeat_penalty = 1.0;
//    int num_threads = 0;
//    bool verbose = false;
//};
struct Args {
    std::string model_path = "..\\..\\..\\..\\..\\model\\GPTQ_INT4-FP16\\";
    std::string device = "GPU";
    InferenceMode mode = INFERENCE_MODE_CHAT;
    bool sync = false;
    std::string prompt = "hello";
    std::string system = "";
    int max_length = 2048;
    int max_new_tokens = -1;
    int max_context_length = 512;
    bool interactive = true;
    //bool convert_kv_fp16 = false;
    bool reduce_logits = false;
    bool do_sample = false;
    int top_k = 0;
    float top_p = 0.7;
    float temp = 0.95;
    float repeat_penalty = 1.0;
    int num_threads = 0;
    bool verbose =  true;
    int output_fixed_len = 0;
};
//static void usage(const std::string &prog) {
//    std::cout << "Usage: " << prog << " [options]\n"
//              << "\n"
//              << "options:\n"
//              << "  -h, --help              show this help message and exit\n"
//              << "  -m, --model PATH        model path (default: chatglm-ggml.bin)\n"
//              << "  --mode                  inference mode chosen from {chat, generate} (default: chat)\n"
//              << "  --sync                  synchronized generation without streaming\n"
//              << "  -p, --prompt PROMPT     prompt to start generation with (default: 你好)\n"
//              << "  --pp, --prompt_path     path to the plain text file that stores the prompt\n"
//              << "  -s, --system SYSTEM     system message to set the behavior of the assistant\n"
//              << "  --sp, --system_path     path to the plain text file that stores the system message\n"
//              << "  -i, --interactive       run in interactive mode\n"
//              << "  -l, --max_length N      max total length including prompt and output (default: 2048)\n"
//              << "  --max_new_tokens N      max number of tokens to generate, ignoring the number of prompt tokens\n"
//              << "  -c, --max_context_length N\n"
//              << "                          max context length (default: 512)\n"
//              << "  --top_k N               top-k sampling (default: 0)\n"
//              << "  --top_p N               top-p sampling (default: 0.7)\n"
//              << "  --temp N                temperature (default: 0.95)\n"
//              << "  --repeat_penalty N      penalize repeat sequence of tokens (default: 1.0, 1.0 = disabled)\n"
//              << "  -t, --threads N         number of threads for inference\n"
//              << "  -v, --verbose           display verbose output including config/system/performance info\n";
//}
static void usage(const std::string& prog) {
    std::cout << "Usage: " << prog << " [options]\n"
        << "\n"
        << "options:\n"
        << "  -h, --help              show this help message and exit\n"
        << "  -m, --model PATH        Chatglm OpenVINO model path (default: modified_openvino_model.xml)\n"
        << "  -token PATH             Tokenizer model path (default: openvino_tokenizer.xml)\n"
        << "  -detoken PATH           DeTokenizer model path (default: openvino_detokenizer.xml)\n"
        << "  -d, --device            Device (default: GPU)\n"
        //<< "  --convert_kv_fp16       Convert kvcache fp16 (default: False)\n"
        << "  --reduce_logits         Reduce_logits (default: False)\n"
        << "  --do_sample             Search (default: False)\n"
        << "  --top_k N               top-k sampling (default: 0)\n"
        << "  --top_p N               top-p sampling (default: 0.7)\n"
        << "  --temp N                temperature (default: 0.95)\n"
        << "  --repeat_penalty N      penalize repeat sequence of tokens (default: 1.0, 1.0 = disabled)\n"
        << "  --output_fixed_len N    set output fixed lenth (default: 0, output lenth is determined by the model)\n";
}
static std::string read_text(std::string path) {
    std::ifstream fin(path);
    CHATGLM_CHECK(fin) << "cannot open file " << path;
    std::ostringstream oss;
    oss << fin.rdbuf();
    return oss.str();
}

//static Args parse_args(const std::vector<std::string> &argv) {
//    Args args;
//
//    for (size_t i = 1; i < argv.size(); i++) {
//        const std::string &arg = argv.at(i);
//
//        if (arg == "-h" || arg == "--help") {
//            usage(argv.at(0));
//            exit(EXIT_SUCCESS);
//        } else if (arg == "-m" || arg == "--model") {
//            args.model_path = argv.at(++i);
//        } else if (arg == "--mode") {
//            args.mode = to_inference_mode(argv.at(++i));
//        } else if (arg == "--sync") {
//            args.sync = true;
//        } else if (arg == "-p" || arg == "--prompt") {
//            args.prompt = argv.at(++i);
//        } else if (arg == "--pp" || arg == "--prompt_path") {
//            args.prompt = read_text(argv.at(++i));
//        } else if (arg == "-s" || arg == "--system") {
//            args.system = argv.at(++i);
//        } else if (arg == "--sp" || arg == "--system_path") {
//            args.system = read_text(argv.at(++i));
//        } else if (arg == "-i" || arg == "--interactive") {
//            args.interactive = true;
//        } else if (arg == "-l" || arg == "--max_length") {
//            args.max_length = std::stoi(argv.at(++i));
//        } else if (arg == "--max_new_tokens") {
//            args.max_new_tokens = std::stoi(argv.at(++i));
//        } else if (arg == "-c" || arg == "--max_context_length") {
//            args.max_context_length = std::stoi(argv.at(++i));
//        } else if (arg == "--top_k") {
//            args.top_k = std::stoi(argv.at(++i));
//        } else if (arg == "--top_p") {
//            args.top_p = std::stof(argv.at(++i));
//        } else if (arg == "--temp") {
//            args.temp = std::stof(argv.at(++i));
//        } else if (arg == "--repeat_penalty") {
//            args.repeat_penalty = std::stof(argv.at(++i));
//        } else if (arg == "-t" || arg == "--threads") {
//            args.num_threads = std::stoi(argv.at(++i));
//        } else if (arg == "-v" || arg == "--verbose") {
//            args.verbose = true;
//        } else {
//            std::cerr << "Unknown argument: " << arg << std::endl;
//            usage(argv.at(0));
//            exit(EXIT_FAILURE);
//        }
//    }
//
//    return args;
//}
static Args parse_args(const std::vector<std::string>& argv) {
    Args args;

    for (size_t i = 1; i < argv.size(); i++) {
        const std::string& arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        }
        else if (arg == "-m" || arg == "--model") {
            args.model_path = argv[++i];
        }
        else if (arg == "-d" || arg == "--device") {
            args.device = argv[++i];
        }
        //else if (arg == "--convert_kv_fp16") {
        //    args.convert_kv_fp16 = true;
        else if (arg == "--reduce_logits") {
            args.reduce_logits = true;
        }
        else if (arg == "--do_sample") {
            args.do_sample = true;
        }
        else if (arg == "--top_k") {
            args.top_k = std::stoi(argv[++i]);
        }
        else if (arg == "--top_p") {
            args.top_p = std::stof(argv[++i]);
        }
        else if (arg == "--temp") {
            args.temp = std::stof(argv[++i]);
        }
        else if (arg == "--repeat_penalty") {
            args.repeat_penalty = std::stof(argv[++i]);
        }
        else if (arg == "--output_fixed_len") {
            args.output_fixed_len = std::stoi(argv[++i]);
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    return args;
}
static Args parse_args(int argc, char **argv) {
    std::vector<std::string> argv_vec;
    argv_vec.reserve(argc);

#ifdef _WIN32
    LPWSTR *wargs = CommandLineToArgvW(GetCommandLineW(), &argc);
    CHATGLM_CHECK(wargs) << "failed to retrieve command line arguments";

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    for (int i = 0; i < argc; i++) {
        argv_vec.emplace_back(converter.to_bytes(wargs[i]));
    }

    LocalFree(wargs);
#else
    for (int i = 0; i < argc; i++) {
        argv_vec.emplace_back(argv[i]);
    }
#endif

    return parse_args(argv_vec);
}

static bool get_utf8_line(std::string &line) {
#ifdef _WIN32
    std::wstring wline;
    bool ret = !!std::getline(std::wcin, wline);
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    line = converter.to_bytes(wline);
    return ret;
#else
    return !!std::getline(std::cin, line);
#endif
}
static double get_duration_ms_until_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<ns>(Time::now() - startTime).count() * 0.000001;
}
static inline void print_message(const chatglm::ChatMessage &message) {
    std::cout << message.content << "\n";
    if (!message.tool_calls.empty() && message.tool_calls.front().type == chatglm::ToolCallMessage::TYPE_CODE) {
        std::cout << message.tool_calls.front().code.input << "\n";
    }
}

static void chat(Args &args) {
    //ggml_time_init();
    //int64_t start_load_us = ggml_time_us();
    auto startTime = Time::now();
    chatglm::Pipeline pipeline(args.model_path, args.device);
    //int64_t end_load_us = ggml_time_us();
    std::string model_name = "ChatGLM3";    // 获取模型的类型名称

    // 创建文本流和性能统计流
    auto text_streamer = std::make_shared<chatglm::TextStreamer>(std::cout, pipeline.chatglm3_tokenizer.get());
    auto perf_streamer = std::make_shared<chatglm::PerfStreamer>();
    // 创建流处理器组，将文本流和性能统计流组合在一起
    std::vector<std::shared_ptr<chatglm::BaseStreamer>> streamers{perf_streamer};
    if (!args.sync) {
        streamers.emplace_back(text_streamer);
    }
    auto streamer = std::make_unique<chatglm::StreamerGroup>(std::move(streamers));
    //streamer = nullptr;
    // 配置生成参数
    chatglm::GenerationConfig gen_config(args.max_length, args.max_new_tokens, args.max_context_length, args.temp > 0,
                                         args.top_k, args.top_p, args.temp, args.repeat_penalty, args.num_threads);
    auto duration_ms = get_duration_ms_until_now(startTime);
    if (args.verbose) {
        /*
        std::cout << "system info: | "
                  << "AVX = " << ggml_cpu_has_avx() << " | "
                  << "AVX2 = " << ggml_cpu_has_avx2() << " | "
                  << "AVX512 = " << ggml_cpu_has_avx512() << " | "
                  << "AVX512_VBMI = " << ggml_cpu_has_avx512_vbmi() << " | "
                  << "AVX512_VNNI = " << ggml_cpu_has_avx512_vnni() << " | "
                  << "FMA = " << ggml_cpu_has_fma() << " | "
                  << "NEON = " << ggml_cpu_has_neon() << " | "
                  << "ARM_FMA = " << ggml_cpu_has_arm_fma() << " | "
                  << "F16C = " << ggml_cpu_has_f16c() << " | "
                  << "FP16_VA = " << ggml_cpu_has_fp16_va() << " | "
                  << "WASM_SIMD = " << ggml_cpu_has_wasm_simd() << " | "
                  << "BLAS = " << ggml_cpu_has_blas() << " | "
                  << "SSE3 = " << ggml_cpu_has_sse3() << " | "
                  << "VSX = " << ggml_cpu_has_vsx() << " |\n";
         */
        std::cout << "inference config: | "
                  << "max_length = " << args.max_length << " | "
                  << "max_context_length = " << args.max_context_length << " | "
                  << "top_k = " << args.top_k << " | "
                  << "top_p = " << args.top_p << " | "
                  << "temperature = " << args.temp << " | "
                  << "repetition_penalty = " << args.repeat_penalty << " | "
                  << "num_threads = " << args.num_threads << " |\n";

        std::cout << "loaded " << model_name << " model from " << args.model_path
                  << " within: " << duration_ms << " ms\n";

        std::cout << std::endl;
    }
    // 检查是否启用了交互模式且推断模式不是 chat
    if (args.mode != INFERENCE_MODE_CHAT && args.interactive) {
        // 如果交互模式启用，但推断模式不是 chat，给出警告
        std::cerr << "interactive demo is only supported for chat mode, falling back to non-interactive one\n";
        args.interactive = false;
    }
    // 准备系统消息
    std::vector<chatglm::ChatMessage> system_messages;
    if (!args.system.empty()) {
        // 如果指定了系统消息，加入系统消息列表
        system_messages.emplace_back(chatglm::ChatMessage::ROLE_SYSTEM, args.system);
    }
    // 如果启用了交互模式
    if (args.interactive) {
        std::cout << R"(    ________          __  ________    __  ___                 )" << '\n'
                  << R"(   / ____/ /_  ____ _/ /_/ ____/ /   /  |/  /_________  ____  )" << '\n'
                  << R"(  / /   / __ \/ __ `/ __/ / __/ /   / /|_/ // ___/ __ \/ __ \ )" << '\n'
                  << R"( / /___/ / / / /_/ / /_/ /_/ / /___/ /  / // /__/ /_/ / /_/ / )" << '\n'
                  << R"( \____/_/ /_/\__,_/\__/\____/_____/_/  /_(_)___/ .___/ .___/  )" << '\n'
                  << R"(                                              /_/   /_/       )" << '\n'
                  << '\n';

        std::cout
            << "Welcome to ChatGLM.OpenVINO! Ask whatever you want. Type 'clear' to clear context. Type 'stop' to exit.\n"
            << "\n";
        // 准备消息列表，包含系统消息
        std::vector<chatglm::ChatMessage> messages = system_messages;
        if (!args.system.empty()) {
            std::cout << std::setw(model_name.size()) << std::left << "System"
                      << " > " << args.system << std::endl;
        }
        // 进入对话循环
        while (1) {
            // 确定当前角色（用户或观察者）
            std::string role;
            if (!messages.empty() && !messages.back().tool_calls.empty()) {
                // 如果有工具调用，取最后一条消息的工具调用类型
                const auto &tool_call = messages.back().tool_calls.front();
                if (tool_call.type == chatglm::ToolCallMessage::TYPE_FUNCTION) {
                    // 如果是函数调用，打印提示信息
                    std::cout << "Function Call > Please manually call function `" << tool_call.function.name
                              << "` with args `" << tool_call.function.arguments << "` and provide the results below.\n"
                              << "Observation   > " << std::flush;
                } else if (tool_call.type == chatglm::ToolCallMessage::TYPE_CODE) {
                    // 如果是代码解释器调用，打印提示信息
                    std::cout << "Code Interpreter > Please manually run the code and provide the results below.\n"
                              << "Observation      > " << std::flush;
                } else {
                    // 未知的工具调用类型，抛出异常
                    CHATGLM_THROW << "unexpected tool type " << tool_call.type;
                }
                // 设置当前角色为观察者
                role = chatglm::ChatMessage::ROLE_OBSERVATION;
            } else {
                // 没有工具调用，当前角色为用户
                // 打印提示信息
                std::cout << std::setw(model_name.size()) << std::left << "Prompt"
                          << " > " << std::flush;
                role = chatglm::ChatMessage::ROLE_USER;
            }
            // 获取用户输入
            std::string prompt;
            if (!get_utf8_line(prompt) || prompt == "stop") {
                break;
            }
            // 处理特殊命令
            if (prompt.empty()) {
                continue;
            }
            if (prompt == "clear") {
                // 清除对话历史
                messages = system_messages;
                continue;
            }
            // 将用户输入作为消息加入消息列表
            messages.emplace_back(std::move(role), std::move(prompt));
            // 打印模型名称
            std::cout << model_name << " > ";
            // 执行 ChatGLM 的 chat 操作
            chatglm::ChatMessage output = pipeline.chat(messages, gen_config, streamer.get());
            // 如果是同步生成模式，打印生成的消息
            if (args.sync) {
                print_message(output);
            }
            // 将生成的消息加入消息列表
            messages.emplace_back(std::move(output));
            // 如果启用了 verbose 模式，打印性能信息
            if (args.verbose) {
                std::cout << "\n" << perf_streamer->to_string() << "\n\n";
            }
            // 重置性能统计信息
            perf_streamer->reset();
        }
        // 打印结束信息
        std::cout << "Bye\n";
    } else {
        // 非交互模式
        if (args.mode == INFERENCE_MODE_CHAT) {
            // 如果推断模式是 chat
            // 准备消息列表，包含系统消息
            std::vector<chatglm::ChatMessage> messages = system_messages;
            // 将用户指定的提示作为消息加入列表
            messages.emplace_back(chatglm::ChatMessage::ROLE_USER, args.prompt);
            // 执行 ChatGLM 的 chat 操作
            chatglm::ChatMessage output = pipeline.chat(messages, gen_config, streamer.get());
            // 如果是同步生成模式，打印生成的消息
            if (args.sync) {
                print_message(output);
            }
        } else {
            // 如果推断模式是 generate
            // 使用 ChatGLM 的 generate 操作生成文本
            std::string output = pipeline.generate(args.prompt, gen_config, streamer.get());
            // 如果是同步生成模式，打印生成的文本
            if (args.sync) {
                std::cout << output << "\n";
            }
        }
        // 如果启用了 verbose 模式，打印性能信息
        if (args.verbose) {
            std::cout << "\n" << perf_streamer->to_string() << "\n\n";
        }
    }
}

int main(int argc, char **argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    _setmode(_fileno(stdin), _O_WTEXT);
#endif    
    // Setting an environment variable
    //if (!SetEnvironmentVariable(L"OV_GPU_MemPreallocationOptions", L"20 65536 2 1.0")) {
    //    std::cerr << "Failed to set environment variable." << std::endl;
    //    return 1;
    //}

    //// Retrieving and printing the environment variable to verify
    //wchar_t buffer[128];
    //if (GetEnvironmentVariable(L"OV_GPU_MemPreallocationOptions", buffer, 128) > 0) {
    //    std::wcout << "OV_GPU_MemPreallocationOptions=" << buffer << std::endl;
    //}
    //else {
    //    std::cerr << "Failed to get environment variable." << std::endl;
    //}
    
    Args args = parse_args(argc, argv);

    try {
        Args args = parse_args(argc, argv);
        chat(args);
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}
