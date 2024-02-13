#include "chatglm.h"
#include <algorithm>
#include <codecvt>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <locale>
#include <numeric>
#include <random>
#include <regex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <filesystem>
#include <openvino/openvino.hpp>
#include <openvino/runtime/properties.hpp>
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/serialize.hpp"
#include "sampling.hpp"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

#ifdef __has_include
#if __has_include(<unistd.h>)
#include <unistd.h>
#if defined(_POSIX_MAPPED_FILES)
#include <sys/mman.h>
#endif
#if defined(_POSIX_MEMLOCK_RANGE)
#include <sys/resource.h>
#endif
#endif
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <stdio.h>
#include <windows.h>
#endif

#ifdef GGML_USE_CUBLAS
#include <ggml-cuda.h>
#endif



namespace chatglm {

    struct UserData { int a = 0; int b = 0; };

void callback(void* userData) {
    UserData* data = reinterpret_cast<UserData*>(userData);
    data->a++;
    data->b++;
    return;
};

//static std::string shape_to_string(ggml_tensor *tensor) {
//    std::ostringstream oss;
//    oss << '[';
//    for (int i = tensor->n_dims - 1; i >= 0; i--) {
//        oss << tensor->ne[i] << (i > 0 ? ", " : "");
//    }
//    oss << ']';
//    return oss.str();
//}
//
//static std::string strides_to_string(ggml_tensor *tensor) {
//    std::ostringstream oss;
//    oss << '[';
//    for (int i = tensor->n_dims - 1; i >= 0; i--) {
//        oss << tensor->nb[i] << (i > 0 ? ", " : "");
//    }
//    oss << ']';
//    return oss.str();
//}
//
//std::string to_string(ggml_tensor *tensor, bool with_data) {
//    std::ostringstream oss;
//    oss << "ggml_tensor(";
//
//    if (with_data) {
//        if (tensor->n_dims > 3)
//            oss << "[";
//        for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
//            if (tensor->n_dims > 2)
//                oss << (i3 > 0 ? ",\n\n[" : "[");
//            for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
//                if (tensor->n_dims > 1)
//                    oss << (i2 > 0 ? ",\n\n[" : "[");
//                for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
//                    oss << (i1 > 0 ? ",\n[" : "[");
//                    for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
//                        auto ptr = (char *)tensor->data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] +
//                                   i0 * tensor->nb[0];
//                        float val;
//                        if (tensor->type == ggml_type_f32) {
//                            val = *(float *)ptr;
//                        } else if (tensor->type == ggml_type_f16) {
//                            val = ggml_fp16_to_fp32(*(ggml_fp16_t *)ptr);
//                        } else {
//                            chatglm_throw << "unimplemented";
//                        }
//                        oss << (i0 > 0 ? ", " : "") << std::setw(7) << std::fixed << std::setprecision(4) << val;
//                    }
//                    oss << "]";
//                }
//                if (tensor->n_dims > 1)
//                    oss << "]";
//            }
//            if (tensor->n_dims > 2)
//                oss << "]";
//        }
//        if (tensor->n_dims > 3)
//            oss << "]";
//        oss << ", ";
//    }
//
//    oss << "shape=" << shape_to_string(tensor) << ", stride=" << strides_to_string(tensor) << ")";
//    return oss.str();
//}
//
//ggml_tensor *tensor_assign_buffers(ggml_tensor *tensor) {
//#ifdef ggml_use_cublas
//    ggml_cuda_assign_buffers(tensor);
//#endif
//    return tensor;
//}
//
//ggml_tensor *tensor_to_device(ggml_tensor *tensor) {
//#ifdef ggml_use_cublas
//    if (tensor->backend == ggml_backend_cpu) {
//        tensor->backend = ggml_backend_gpu;
//        ggml_cuda_transform_tensor(tensor->data, tensor);
//    }
//#endif
//    return tensor;
//}
//
//ggml_tensor *tensor_to_cpu(ggml_tensor *tensor) {
//#ifdef ggml_use_cublas
//    if (tensor->backend != ggml_backend_cpu) {
//        ggml_cuda_free_data(tensor);
//        tensor->backend = ggml_backend_cpu;
//    }
//#endif
//    return tensor;
//}

const std::string ToolCallMessage::TYPE_FUNCTION = "function";
const std::string ToolCallMessage::TYPE_CODE = "code";

const std::string ChatMessage::ROLE_USER = "user";
const std::string ChatMessage::ROLE_ASSISTANT = "assistant";
const std::string ChatMessage::ROLE_SYSTEM = "system";
const std::string ChatMessage::ROLE_OBSERVATION = "observation";

static double get_duration_ms_until_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<ns>(Time::now() - startTime).count() * 0.000001;
}


/*@brief Insert slice transformation matches following graph, start from logits (Results) to search along root->parent-> grandparent node,
 * then insert slice between Reshape (grandparent node) and Matmul to keep only last dim of matmul first input, first input shape reduced
 * from [1, seq_len, 4096] to [1, 1,4096]. Therefore, after graph transformation, we can reduce matmul computation
 * from [1, seq_len, 4096] * [1, 4096, 151936] = [1, seq_len, 151936] to [1,1,4096]*[4096,151936] = [1,1,151936]
 *
 * Original graph
 *         +----------+            +----------+
 *         |  Reshape |            | Constant |
 *         +----------+            +----------+
 *              |                       |
 *              -----------    ----------
 *                        |    |
 *                        v    v
 *                      +--------+
 *                      | MatMul |
 *                      +--------+
 *                          |
 *                          v
 *                     +----------+
 *                     |  logits  |
 *                     +----------+
 *
 * Modified graph after insert slice:
 *
 *         +----------+            +----------+
 *         |  Reshape |            | Constant |
 *         +----------+            +----------+
 *              |                       |
 *         +----------+                 |
 *         |  Slice   |                 |
 *         +----------+                 |
 *              |                       |
 *              -----------    ----------
 *                        |    |
 *                        v    v
 *                      +--------+
 *                      | MatMul |
 *                      +--------+
 *                          |
 *                          v
 *                     +----------+
 *                     |  logits  |
 *                     +----------+
*/

class InsertSlice : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertSlice", "0");
    explicit InsertSlice() {
        auto label = ov::pass::pattern::wrap_type<ov::op::v0::Result>();
        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            auto root = std::dynamic_pointer_cast<ov::op::v0::Result>(m.get_match_root());
            if (!root) {
                return false;
            }
            std::string root_name = root->get_friendly_name();
            if (root->get_output_partial_shape(0).size() == 3) {
                std::cout << "Find target root node name: " << root_name << "\n";
                auto parent = root->input_value(0).get_node_shared_ptr();
                std::cout << "Find parent node name: " << parent->get_friendly_name() << "\n";
                auto grand_parent1 = parent->input_value(0).get_node_shared_ptr();
                std::cout << "Find grandparent1 node name: " << grand_parent1->get_friendly_name() << "\n";

                auto grand_parent = grand_parent1->input_value(0).get_node_shared_ptr();
                std::cout << "Find grandparent node name: " << grand_parent->get_friendly_name() << "\n";

                ov::Output<ov::Node> grand_parent_output = grand_parent1->get_input_source_output(0); // parent->get_input_source_output(0);
                std::set<ov::Input<ov::Node>> consumers = grand_parent_output.get_target_inputs();

                std::vector<int32_t> start_v = { -1, 0, 0 };
                std::vector<int32_t> stop_v = { -2, 1,4096 };
                std::vector<int32_t> step_v = { -1, 1, 1 };

                std::cout << "Original reshape node output shape:" << grand_parent_output.get_partial_shape() << std::endl;
                auto starts = ov::op::v0::Constant::create(ov::element::i32,
                    ov::Shape{ 3 },
                    start_v);
                auto stop = ov::op::v0::Constant::create(ov::element::i32,
                    ov::Shape{ 3 },
                    stop_v);
                auto step = ov::op::v0::Constant::create(ov::element::i32,
                    ov::Shape{ 3 },
                    step_v);
                auto slice = std::make_shared<ov::opset13::Slice>(grand_parent, starts, stop, step); //data, starts, ends, steps
                std::cout << "After insert slice node, output shape" << slice->output(0).get_partial_shape() << std::endl;
                for (auto consumer : consumers) {
                    consumer.replace_source_output(slice->output(0));
                }
                register_new_node(slice);
            }

            return true;
        };
        // Register pattern with Parameter operation as a pattern root node
        auto m = std::make_shared<ov::pass::pattern::Matcher>(label, "InsertSlice");
        // Register Matcher
        register_matcher(m, callback);
    }
};
void BaseTokenizer::check_chat_messages(const std::vector<ChatMessage> &messages) {
    CHATGLM_CHECK(messages.size() % 2 == 1) << "invalid chat messages size " << messages.size();
    for (size_t i = 0; i < messages.size(); i++) {
        const std::string &target_role = (i % 2 == 0) ? ChatMessage::ROLE_USER : ChatMessage::ROLE_ASSISTANT;
        CHATGLM_CHECK(messages[i].role == target_role)
            << "expect messages[" << i << "].role to be " << target_role << ", but got " << messages[i].role;
    }
}

// Adapted from https://github.com/ggerganov/llama.cpp/blob/master/llama.cpp
//void ggml_graph_compute_helper(std::vector<uninitialized_char> &buf, ggml_cgraph *graph, int n_threads) {
//    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);
//
//    if (plan.work_size > 0) {
//        buf.resize(plan.work_size);
//        plan.work_data = (uint8_t *)buf.data();
//    }
//
//    ggml_graph_compute(graph, &plan);
//}

// for debugging purpose
//[[maybe_unused]] static inline ggml_tensor *add_zero(ggml_context *ctx, ggml_tensor *tensor) {
//    ggml_tensor *zeros = ggml_new_tensor(ctx, tensor->type, tensor->n_dims, tensor->ne);
//    ggml_set_f32(zeros, 0);
//    tensor_to_device(zeros);
//    ggml_tensor *out = tensor_assign_buffers(ggml_add(ctx, tensor, zeros));
//    return out;
//}

//void ModelContext::init_device_context() {
//#ifdef GGML_USE_METAL
//    ctx_metal = make_unique_ggml_metal_context(1);
//
//    const size_t max_size = ggml_get_max_tensor_size(ctx_w.get());
//
//    void *weight_data = weight_buffer.empty() ? ggml_get_mem_buffer(ctx_w.get()) : (void *)weight_buffer.data();
//    size_t weight_size = weight_buffer.empty() ? ggml_get_mem_size(ctx_w.get()) : weight_buffer.size();
//    CHATGLM_CHECK(ggml_metal_add_buffer(ctx_metal.get(), "weights", weight_data, weight_size, max_size));
//
//    CHATGLM_CHECK(ggml_metal_add_buffer(ctx_metal.get(), "kv", ggml_get_mem_buffer(ctx_kv.get()),
//                                        ggml_get_mem_size(ctx_kv.get()), 0));
//
//    void *compute_data = ctx_b ? ggml_get_mem_buffer(ctx_b.get()) : compute_buffer.data();
//    size_t compute_size = ctx_b ? ggml_get_mem_size(ctx_b.get()) : compute_buffer.size();
//    CHATGLM_CHECK(ggml_metal_add_buffer(ctx_metal.get(), "compute", compute_data, compute_size, 0));
//
//    CHATGLM_CHECK(ggml_metal_add_buffer(ctx_metal.get(), "scratch", scratch.data, scratch.size, 0));
//#endif
//}

// ===== streamer =====

void StreamerGroup::put(const std::vector<int> &output_ids) {
    for (auto &streamer : streamers_) {
        streamer->put(output_ids);
    }
}

void StreamerGroup::end() {
    for (auto &streamer : streamers_) {
        streamer->end();
    }
}

// reference: https://stackoverflow.com/questions/216823/how-to-trim-a-stdstring

// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    rtrim(s);
    ltrim(s);
}

void TextStreamer::put(const std::vector<int> &output_ids) {
    if (is_prompt_) {
        // skip prompt
        is_prompt_ = false;
        return;
    }
    
    static const std::vector<char> puncts{',', '!', ':', ';', '?'};

    token_cache_.insert(token_cache_.end(), output_ids.begin(), output_ids.end());
    std::string text = tokenizer_->decode(token_cache_);
    
    if (is_first_line_) {
        ltrim(text);
    }
    if (text.empty()) {
        return;
    }

    std::string printable_text;
    if (text.back() == '\n') {
        // flush the cache after newline
        printable_text = text.substr(print_len_);
        is_first_line_ = false;
        token_cache_.clear();
        print_len_ = 0;
    } else if (std::find(puncts.begin(), puncts.end(), text.back()) != puncts.end()) {
        // last symbol is a punctuation, hold on
    } else if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
        // ends with an incomplete token, hold on
    } else {
        printable_text = text.substr(print_len_);
        print_len_ = text.size();
    }    
   
    if (py_streamer) {
        //pycallback(&py_context);        
        py_handler(printable_text, output_ids);
    }
    else
    {
        os_ << printable_text << std::flush;
        
    }
  
}

void TextStreamer::end() {
    std::string text = tokenizer_->decode(token_cache_);
    if (is_first_line_) {
        ltrim(text);
    }
    os_ << text.substr(print_len_) << std::endl;
    is_prompt_ = true;
    is_first_line_ = true;
    token_cache_.clear();
    print_len_ = 0;
    py_streamer = false;    
    py_userData = "";
    py_handler = nullptr;
}
void TextStreamer::registerCallBack(std::function<void(std::string, std::vector<int>)>& handler) {
    py_handler = handler;
}
void PerfStreamer::put(const std::vector<int> &output_ids) {
    CHATGLM_CHECK(!output_ids.empty());
    if (num_prompt_tokens_ == 0) {
        // before prompt eval
        //start_us_ = ggml_time_us();
        //auto currentTime = std::chrono::steady_clock::now();
        //auto duration = currentTime.time_since_epoch();
        //start_us_ = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
        auto currentTime = std::chrono::high_resolution_clock::now();
        start_us_ = std::chrono::duration_cast<std::chrono::microseconds>(currentTime.time_since_epoch()).count();
        num_prompt_tokens_ = output_ids.size();
    } else {
        if (num_output_tokens_ == 0) {
            // first new token
            //prompt_us_ = ggml_time_us();            
/*            auto currentTime = std::chrono::steady_clock::now();
            auto duration = currentTime.time_since_epoch();
            prompt_us_ = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count(); */ 
            auto currentTime = std::chrono::high_resolution_clock::now();
            prompt_us_ = std::chrono::duration_cast<std::chrono::microseconds>(currentTime.time_since_epoch()).count();
        }
        num_output_tokens_ += output_ids.size();
    }
}

void PerfStreamer::reset() {
    start_us_ = prompt_us_ = end_us_ = 0;
    num_prompt_tokens_ = num_output_tokens_ = 0;
}

std::string PerfStreamer::to_string() const {
    std::ostringstream oss;
    oss << "prompt time: " << prompt_total_time_us() / 1000.f << " ms / " << num_prompt_tokens() << " tokens ("
        << prompt_token_time_us() / 1000.f << " ms/token) /" << 1000000.f / prompt_token_time_us() << " token/sec \n"
        << "output time: " << output_total_time_us() / 1000.f << " ms / " << num_output_tokens() << " tokens ("
        << output_token_time_us() / 1000.f << " ms/token) /" << 1000000.f / output_token_time_us() << " token/sec \n"
        << "total time: " << (prompt_total_time_us() + output_total_time_us()) / 1000.f << " ms";
    return oss.str();
}

#ifdef _POSIX_MAPPED_FILES
MappedFile::MappedFile(const std::string &path) {
    int fd = open(path.c_str(), O_RDONLY);
    CHATGLM_CHECK(fd > 0) << "cannot open file " << path << ": " << strerror(errno);

    struct stat sb;
    CHATGLM_CHECK(fstat(fd, &sb) == 0) << strerror(errno);
    size = sb.st_size;

    data = (char *)mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    CHATGLM_CHECK(data != MAP_FAILED) << strerror(errno);

    CHATGLM_CHECK(close(fd) == 0) << strerror(errno);
}

MappedFile::~MappedFile() { CHATGLM_CHECK(munmap(data, size) == 0) << strerror(errno); }
#elif defined(_WIN32)
MappedFile::MappedFile(const std::string &path) {

    //int fd = open(path.c_str(), O_RDONLY);
    int fd = _open(path.c_str(), O_RDONLY);
    CHATGLM_CHECK(fd > 0) << "cannot open file " << path << ": " << strerror(errno);

    struct _stat64 sb;
    CHATGLM_CHECK(_fstat64(fd, &sb) == 0) << strerror(errno);
    size = sb.st_size;

    HANDLE hFile = (HANDLE)_get_osfhandle(fd);

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    CHATGLM_CHECK(hMapping != NULL) << strerror(errno);

    data = (char *)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMapping);

    CHATGLM_CHECK(data != NULL) << strerror(errno);

    //CHATGLM_CHECK(close(fd) == 0) << strerror(errno);
    CHATGLM_CHECK(_close(fd) == 0) << strerror(errno);
}

MappedFile::~MappedFile() { CHATGLM_CHECK(UnmapViewOfFile(data)) << strerror(errno); }
#endif

//void ModelLoader::seek(int64_t offset, int whence) {
//    if (whence == SEEK_SET) {
//        ptr = data + offset;
//    } else if (whence == SEEK_CUR) {
//        ptr += offset;
//    } else if (whence == SEEK_END) {
//        ptr = data + size + offset;
//    } else {
//        CHATGLM_THROW << "invalid seek mode " << whence;
//    }
//}
//
//std::string ModelLoader::read_string(size_t length) {
//    std::string s(ptr, ptr + length);
//    ptr += length;
//    return s;
//}
//
//void ModelLoader::checked_read_tensor_meta(const std::string &name, int target_ndim, int64_t *target_ne,
//                                           ggml_type target_dtype) {
//    // read and check tensor name
//    {
//        int name_size = read_basic<int>();
//        CHATGLM_CHECK(name_size == (int)name.size())
//            << "tensor " << name << " name size mismatch: expect " << name.size() << " but got " << name_size;
//        std::string weight_name = read_string(name_size);
//        CHATGLM_CHECK(weight_name == name) << "tensor name mismatch: expect " << name << " but got " << weight_name;
//    }
//
//    // read and check tensor shape
//    {
//        int ndim = read_basic<int>();
//        CHATGLM_CHECK(ndim == target_ndim)
//            << "tensor " << name << " ndim mismatch: expect " << target_ndim << " but got " << ndim;
//        for (int i = ndim - 1; i >= 0; i--) {
//            int dim_size = read_basic<int>();
//            CHATGLM_CHECK(dim_size == target_ne[i]) << "tensor " << name << " shape mismatch at dim " << i
//                                                    << ": expect " << target_ne[i] << " but got " << dim_size;
//        }
//    }
//
//    // read and check tensor dtype
//    {
//        ggml_type dtype = (ggml_type)read_basic<int>();
//        CHATGLM_CHECK(dtype == target_dtype)
//            << "tensor " << name << " dtype mismatch: expect " << target_dtype << " but got " << dtype;
//    }
//}
//
//void *ModelLoader::read_tensor_data(size_t nbytes) {
//    constexpr int64_t MEM_ALIGNED = 16;
//    const int64_t data_offset = (tell() + (MEM_ALIGNED - 1)) & ~(MEM_ALIGNED - 1);
//    void *tensor_data = data + data_offset;
//    seek(data_offset + nbytes, SEEK_SET);
//    return tensor_data;
//}
//
//void ModelLoader::read_tensor(const std::string &name, ggml_tensor *tensor) {
//    checked_read_tensor_meta(name, tensor->n_dims, tensor->ne, tensor->type);
//    tensor->data = read_tensor_data(ggml_nbytes(tensor));
//}
//
//// ===== modules =====
//
//ggml_tensor *Embedding::forward(ModelContext *ctx, ggml_tensor *input) const {
//    ggml_tensor *output = ggml_get_rows(ctx->ctx_b.get(), weight, input);
//    return output;
//}
//
//ggml_tensor *Linear::forward(ModelContext *ctx, ggml_tensor *input) const {
//    // input: [seqlen, in_features]
//    ggml_context *gctx = ctx->ctx_b.get();
//    ggml_tensor *output = tensor_assign_buffers(ggml_mul_mat(gctx, weight, input)); // [seqlen, out_features]
//    if (bias) {
//        output = tensor_assign_buffers(ggml_add_inplace(gctx, output, bias));
//    }
//    return output;
//}
//
//ggml_tensor *LayerNorm::forward(ModelContext *ctx, ggml_tensor *input) const {
//    // input: [seqlen, normalized_shape]
//    ggml_context *gctx = ctx->ctx_b.get();
//    auto ggml_norm_fn = inplace ? ggml_norm_inplace : ggml_norm;
//    ggml_tensor *output = tensor_assign_buffers(ggml_norm_fn(gctx, input, eps));
//    output = tensor_assign_buffers(ggml_mul_inplace(gctx, output, weight));
//    output = tensor_assign_buffers(ggml_add_inplace(gctx, output, bias));
//    return output;
//}
//
//ggml_tensor *RMSNorm::forward(ModelContext *ctx, ggml_tensor *input) const {
//    ggml_context *gctx = ctx->ctx_b.get();
//    auto ggml_rms_norm_fn = inplace ? ggml_rms_norm_inplace : ggml_rms_norm;
//    ggml_tensor *output = tensor_assign_buffers(ggml_rms_norm_fn(gctx, input, eps));
//    output = tensor_assign_buffers(ggml_mul_inplace(gctx, output, weight));
//    return output;
//}

// Adapted from https://github.com/ggerganov/llama.cpp/blob/master/examples/common.cpp
int get_num_physical_cores() {
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

int get_default_num_threads() {
#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_METAL)
    return 1;
#else
    return std::min(get_num_physical_cores(), 16);
#endif
}

std::string to_string(ModelType model_type) {
    switch (model_type) {
    case ModelType::CHATGLM:
        return "ChatGLM";
    case ModelType::CHATGLM2:
        return "ChatGLM2";
    case ModelType::CHATGLM3:
        return "ChatGLM3";
    case ModelType::BAICHUAN7B:
        return "Baichuan7B";
    case ModelType::BAICHUAN13B:
        return "Baichuan13B";
    case ModelType::INTERNLM:
        return "InternLM";
    default:
        CHATGLM_THROW << "unknown model type " << (int)model_type;
    }
}

BaseModelForCausalLM::BaseModelForCausalLM(ov::Core core, const std::string openvino_model_path, const std::string device, ov::AnyMap device_config)
    : config(config) {
    
    ov::CompiledModel compilemodel = core.compile_model(openvino_model_path, device, device_config);
    ireq = compilemodel.create_infer_request();
    model_inputs = compilemodel.inputs();
    // load config

    config.model_type = ModelType::CHATGLM3;
    config.max_length = 8192;
    config.bos_token_id = -1;
    config.eos_token_id = 2;
    config.pad_token_id = 0;
    config.sep_token_id = -1;
    // 初始化上下文（context）对象
//    ctx_.dtype = config.dtype;
//
//    // 计算权重上下文的大小
//    const size_t ctx_w_size = num_weights * ggml_tensor_overhead();
//
//    // 计算键值对（key-value pairs）上下文的大小
//    const size_t ctx_kv_size = 2 * config.num_hidden_layers *
//        (config.max_length * config.hidden_size / config.num_attention_heads *
//            config.num_kv_heads * ggml_type_size(GGML_TYPE_F16) +
//            ggml_tensor_overhead());
//
//    // 创建权重上下文
//    ctx_.ctx_w = make_unique_ggml_context(ctx_w_size, nullptr, true);
//
//    // 创建键值对上下文，额外分配1MB的空间
//    ctx_.ctx_kv = make_unique_ggml_context(ctx_kv_size + 1 * MB, nullptr, false);
//
//    // 额外的1MB空间用于MPS（Memory Protection Services）
//    ctx_.compute_buffer.resize(mem_size);
//    ctx_.scratch_buffer.resize(scratch_size);
//
//    // 设置上下文的初始状态
//    ctx_.scratch = { 0, ctx_.scratch_buffer.size(), ctx_.scratch_buffer.data() };
//
//    // 根据是否使用CUBLAS设置CUDA的额外scratch大小
//#ifdef GGML_USE_CUBLAS
//    ggml_cuda_set_scratch_size(scratch_size);
//#endif
}

int64_t BaseModelForCausalLM::get_out_token_id(const std::vector<int>& input_ids, float* logits, size_t vocab_size, const GenerationConfig& gen_config) {
    int64_t out_token;

    // logits pre-process
    if (gen_config.repetition_penalty != 1.f) {
        sampling_repetition_penalty(logits, logits + vocab_size, input_ids, gen_config.repetition_penalty);
    }

    if (gen_config.do_sample)
    {
        if (gen_config.temperature > 0) {
            sampling_temperature(logits, logits + vocab_size, gen_config.temperature);
        }

        std::vector<TokenIdScore> token_scores(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            token_scores[i] = TokenIdScore(i, logits[i]);
        }

        // top_k sampling
        if (0 < gen_config.top_k && gen_config.top_k < (int)token_scores.size()) {
            sampling_top_k(token_scores.data(), token_scores.data() + gen_config.top_k,
                token_scores.data() + token_scores.size());
            token_scores.resize(gen_config.top_k);
        }

        // top_p sampling
        if (0.f < gen_config.top_p && gen_config.top_p < 1.f) {
            auto pos = sampling_top_p(token_scores.data(), token_scores.data() + token_scores.size(), gen_config.top_p);
            token_scores.resize(pos - token_scores.data());
        }

        // sample next token
        sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());
        for (size_t i = 0; i < token_scores.size(); i++) {
            logits[i] = token_scores[i].score;
        }

        thread_local std::random_device rd;
        thread_local std::mt19937 gen(rd());

        std::discrete_distribution<> dist(logits, logits + token_scores.size());
        out_token = token_scores[dist(gen)].id;
    }
    else {
        out_token = std::max_element(logits, logits + vocab_size) - logits;
    }

    return out_token;
}

int BaseModelForCausalLM::generate_next_token(const ov::Tensor input_ids, const GenerationConfig& gen_config,
    int n_past, int n_ctx) {
    // 创建ggml_context和ggml_graph
    //ctx_.ctx_b = make_unique_ggml_context(ctx_.compute_buffer.size(), ctx_.compute_buffer.data(), false);
    //ctx_.gf = {};

    // 设置线程数
    //int n_threads = gen_config.num_threads; // 用户定义
    //if (n_threads <= 0) {
    //    n_threads = get_default_num_threads(); // 默认线程数
    //}

    // 根据条件设置线程数
    //int curr_input_ids_size = input_ids.size() - n_past;
    //if (curr_input_ids_size >= 32 && ggml_cpu_has_blas() && !ggml_cpu_has_gpublas()) {
    //    n_threads = 1; // 如果启用了BLAS，则使用1个线程
    //}

    // 创建ggml_tensor，复制当前输入令牌
    //ggml_tensor* curr_input_ids = ggml_new_tensor_1d(ctx_.ctx_b.get(), GGML_TYPE_I32, curr_input_ids_size);
    //memcpy(curr_input_ids->data, input_ids.data() + n_past, ggml_nbytes(curr_input_ids));

    // 进行前向推理，获取语言模型的logits
    //ggml_tensor* lm_logits = forward(&ctx_, curr_input_ids, n_past, n_ctx);
    //lm_logits->backend = GGML_BACKEND_CPU;

    // 构建前向图
    //ggml_build_forward_expand(&ctx_.gf, lm_logits);

//#ifdef GGML_USE_METAL
//    ggml_metal_graph_compute(ctx_.ctx_metal.get(), &ctx_.gf);
//#else
//    ggml_graph_compute_helper(ctx_.work_buffer, &ctx_.gf, n_threads);
//#endif
//
//#ifdef GGML_PERF
//    ggml_graph_print(&ctx_.gf);
//#endif








    // 获取下一个输出令牌
    int out_token;

    //// 检查是否存在NaN或Inf
    //for (int i = 0; i < vocab_size; i++) {
    //    CHATGLM_CHECK(std::isfinite(logits[i])) << "nan/inf encountered at lm_logits[" << i << "]";
    //}

    //// 对logits进行预处理
    //if (gen_config.repetition_penalty != 1.f) {
    //    sampling_repetition_penalty(logits, logits + vocab_size, input_ids,
    //        gen_config.repetition_penalty);
    //}

    //int next_token_id;
    //if (gen_config.do_sample) {
    //    // temperature sampling
    //    if (gen_config.temperature > 0) {
    //        sampling_temperature(logits, logits + vocab_size, gen_config.temperature);
    //    }

    //    // 构建TokenIdScore向量，存储每个令牌的分数
    //    std::vector<TokenIdScore> token_scores(vocab_size);
    //    for (int i = 0; i < vocab_size; i++) {
    //        token_scores[i] = TokenIdScore(i, logits[i]);
    //    }

    //    // top_k sampling
    //    if (0 < gen_config.top_k && gen_config.top_k < (int)token_scores.size()) {
    //        sampling_top_k(token_scores.data(), token_scores.data() + gen_config.top_k,
    //            token_scores.data() + token_scores.size());
    //        token_scores.resize(gen_config.top_k);
    //    }

    //    // top_p sampling
    //    if (0.f < gen_config.top_p && gen_config.top_p < 1.f) {
    //        auto pos = sampling_top_p(token_scores.data(), token_scores.data() + token_scores.size(), gen_config.top_p);
    //        token_scores.resize(pos - token_scores.data());
    //    }

    //    // 对分数进行softmax处理
    //    sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());

    //    // 使用离散分布进行采样，得到下一个令牌的ID
    //    thread_local std::random_device rd;
    //    thread_local std::mt19937 gen(rd());
    //    std::discrete_distribution<> dist(next_token_logits, next_token_logits + token_scores.size());
    //    next_token_id = token_scores[dist(gen)].id;
    //}
    //else {
    //    // 贪婪搜索，选择分数最高的令牌
    //    next_token_id = std::max_element(next_token_logits, next_token_logits + vocab_size) - next_token_logits;
    //}

    return out_token;
}

void BaseModelForCausalLM::sampling_repetition_penalty(float *first, float *last, const std::vector<int>& input_ids,
                                                       float penalty) {
    CHATGLM_CHECK(penalty > 0) << "penalty must be a positive float, but got " << penalty;
    const float inv_penalty = 1.f / penalty;
    const int vocab_size = last - first;
    std::vector<bool> occurrence(vocab_size, false);
    for (const int id : input_ids) {
        if (!occurrence[id]) {
            first[id] *= (first[id] > 0) ? inv_penalty : penalty;
        }
        occurrence[id] = true;
    }
}

void BaseModelForCausalLM::sampling_temperature(float *first, float *last, float temp) {
    const float inv_temp = 1.f / temp;
    for (float *it = first; it != last; it++) {
        *it *= inv_temp;
    }
}

void BaseModelForCausalLM::sampling_top_k(TokenIdScore *first, TokenIdScore *kth, TokenIdScore *last) {
    std::nth_element(first, kth, last, std::greater<TokenIdScore>());
}

TokenIdScore *BaseModelForCausalLM::sampling_top_p(TokenIdScore *first, TokenIdScore *last, float top_p) {
    // fast top_p in expected O(n) time complexity
    sampling_softmax_inplace(first, last);

    while (first + 1 < last) {
        const float pivot_score = (last - 1)->score; // use mid score?
        TokenIdScore *mid =
            std::partition(first, last - 1, [pivot_score](const TokenIdScore &x) { return x.score > pivot_score; });
        std::swap(*mid, *(last - 1));

        const float prefix_sum =
            std::accumulate(first, mid, 0.f, [](float sum, const TokenIdScore &x) { return sum + x.score; });
        if (prefix_sum >= top_p) {
            last = mid;
        } else if (prefix_sum + mid->score < top_p) {
            first = mid + 1;
            top_p -= prefix_sum + mid->score;
        } else {
            return mid + 1;
        }
    }
    return last;
}

void BaseModelForCausalLM::sampling_softmax_inplace(TokenIdScore *first, TokenIdScore *last) {
    float max_score = std::max_element(first, last)->score;
    float sum = 0.f;
    for (TokenIdScore *p = first; p != last; p++) {
        float s = std::exp(p->score - max_score);
        p->score = s;
        sum += s;
    }
    float inv_sum = 1.f / sum;
    for (TokenIdScore *p = first; p != last; p++) {
        p->score *= inv_sum;
    }
}
// 生成函数，输入参数包括输入向量（input_ids）、生成配置（gen_config）和用于流式处理的基类指针（streamer）
std::vector<int> BaseModelForCausalLM::generate(const ov::Tensor input_ids, const ov::Tensor attention_mask, const GenerationConfig &gen_config,
                                                BaseStreamer *streamer) {
    // 检查生成配置中的最大长度是否小于等于模型的最大长度
    CHATGLM_CHECK(gen_config.max_length <= config.max_length)
        << "requested max_length (" << gen_config.max_length << ") is larger than model's max_length ("
        << config.max_length << ")";
    // 初始化输出向量，并预留足够的空间以容纳生成的文本
    std::vector<int> output_ids;
    constexpr size_t BATCH_SIZE = 1;
    output_ids.reserve(gen_config.max_length);
    // 将输入文本拷贝到输出文本中
    for (size_t idx = 0; idx < input_ids.get_size(); ++idx) {
        output_ids.emplace_back(((int)input_ids.data<const int64_t>()[idx]));
    }
    //output_ids = input_ids;// 将输入文本拷贝到输出文本中

    if (streamer) {
        streamer->put(output_ids);// 如果提供了流式处理器，则将输入文本放入流中
    }

    /* 配置输入张量的形状
    for (size_t idx = 3; idx < model_inputs.size(); ++idx) {
        ov::PartialShape shape = model_inputs.at(idx).get_partial_shape().get_min_shape();
        shape[1] = BATCH_SIZE;
        ireq.get_input_tensor(idx).set_shape(shape.get_shape());
    }
    ireq.get_tensor("input_ids").set_shape(input_ids.get_shape());
    ireq.get_tensor("attention_mask").set_shape(attention_mask.get_shape());
    std::copy_n(input_ids.data<const int64_t>(), input_ids.get_size(), ireq.get_tensor("input_ids").data<int64_t>());
    std::fill_n(ireq.get_tensor("attention_mask").data<int64_t>(), attention_mask.get_size(), 1);*/
    ireq.set_tensor("input_ids", input_ids);
    ireq.set_tensor("attention_mask", attention_mask);    

    ireq.get_tensor("position_ids").set_shape(input_ids.get_shape());
    std::iota(ireq.get_tensor("position_ids").data<int64_t>(), ireq.get_tensor("position_ids").data<int64_t>() + ireq.get_tensor("position_ids").get_size(), 0);
    ireq.get_tensor("beam_idx").set_shape({ BATCH_SIZE });
    ireq.get_tensor("beam_idx").data<int32_t>()[0] = 0;
    
    for (auto&& state : ireq.query_state()) {
        state.reset();
    }
    // 进行第一个令牌的推理
    ireq.infer();
    //获取logits张量的最后一个维度，即词汇表大小。
    size_t vocab_size = ireq.get_tensor("logits").get_shape().back();
    float* logits = ireq.get_tensor("logits").data<float>();
    //float* logits = ireq.get_tensor("logits").data<float>() + (input_ids.get_size() - 1) * vocab_size;
    // 获取下一个输出令牌
    int64_t next_token_id = get_out_token_id(output_ids, logits, vocab_size, gen_config);
    output_ids.emplace_back((int)next_token_id);
    
    if (streamer) {
        streamer->put({ (int)next_token_id });// 如果提供了流式处理器，则将输入文本放入流中
    }
    ireq.get_tensor("input_ids").set_shape({ BATCH_SIZE, 1 });
    ireq.get_tensor("position_ids").set_shape({ BATCH_SIZE, 1 });

    int n_past = 0;// 过去文本的起始索引
    const int n_ctx = input_ids.get_size();// 输入文本的上下文大小
    const int max_new_tokens = (gen_config.max_new_tokens > 0) ? gen_config.max_new_tokens : gen_config.max_length;
    
    // 生成文本，直到达到最大长度或生成了足够的新令牌
    while (output_ids.size() < std::min(gen_config.max_length, n_ctx + max_new_tokens)) {
        ireq.get_tensor("input_ids").data<int64_t>()[0] = next_token_id;
        ireq.get_tensor("attention_mask").set_shape({ BATCH_SIZE, ireq.get_tensor("attention_mask").get_shape()[1] + 1 });
        std::fill_n(ireq.get_tensor("attention_mask").data<int64_t>(), ireq.get_tensor("attention_mask").get_size(), 1);
        ireq.get_tensor("position_ids").data<int64_t>()[0] = ireq.get_tensor("attention_mask").get_size() - 2;
        //for (size_t idx = 3; idx < model_inputs.size(); ++idx) {
        //    ireq.set_input_tensor(idx, ireq.get_output_tensor(idx - 2));
        //}
        ireq.start_async();
        ireq.wait();                  
        logits = ireq.get_tensor("logits").data<float>();

        // 生成下一个令牌
        next_token_id = get_out_token_id(output_ids, logits, vocab_size, gen_config);

        n_past = output_ids.size();// 更新过去文本的起始索引
        output_ids.emplace_back((int)next_token_id);// 将生成的令牌添加到输出文本中

        if (streamer) {
            streamer->put({ (int)next_token_id});// 如果提供了流式处理器，则将生成的令牌放入流中
        }
        // 如果生成了结束符或额外的结束符，则停止生成
        if (next_token_id == config.eos_token_id ||
            std::find(config.extra_eos_token_ids.begin(), config.extra_eos_token_ids.end(), next_token_id) !=
            config.extra_eos_token_ids.end()) {
            //std::cout << "\n received eos token " << next_token_id << std::endl;
            break;
        }
    }
    // 结束流式处理
    if (streamer) {
        streamer->end();
    }
    // 返回生成的文本序列
    return output_ids;
}

// ===== ChatGLM3-6B =====

ChatGLM3Tokenizer::ChatGLM3Tokenizer(ov::Core core, const std::string& tokenizer_path, const std::string& detokenizer_path)

{
    //const auto status = sp.LoadFromSerializedProto(serialized_model_proto);
    //CHATGLM_CHECK(status.ok()) << status.ToString();
    
    //core.add_extension(USER_OV_EXTENSIONS_PATH);  // USER_OV_EXTENSIONS_PATH is defined in root CMakeLists.txt

    ov_tokenizer = core.compile_model(tokenizer_path, "CPU").create_infer_request();
    ov_detokenizer = core.compile_model(detokenizer_path, "CPU").create_infer_request();
    
    int special_id = 64789; // There's no way to extract the value from the detokenizer for now by OpenVINO
    mask_token_id = special_id++;
    gmask_token_id = special_id++;
    smask_token_id = special_id++;
    sop_token_id = special_id++;
    eop_token_id = special_id++;
    system_token_id = special_id++;
    user_token_id = special_id++;
    assistant_token_id = special_id++;
    observation_token_id = special_id++;

    special_tokens = {
        {"[MASK]", mask_token_id},
        {"[gMASK]", gmask_token_id},
        {"[sMASK]", smask_token_id},
        {"sop", sop_token_id},
        {"eop", eop_token_id},
        {"<|system|>", system_token_id},
        {"<|user|>", user_token_id},
        {"<|assistant|>", assistant_token_id},
        {"<|observation|>", observation_token_id},
    };

    for (const auto &item : special_tokens) {
        index_special_tokens[item.second] = item.first;
    }
}

std::pair<ov::Tensor, ov::Tensor> ChatGLM3Tokenizer::tokenize(std::string&& prompt) const {
    constexpr size_t BATCH_SIZE = 1;
    //ov::Tensor destination = ov_tokenizer.get_input_tensor();
    //openvino_extensions::pack_strings(std::array<std::string_view, BATCH_SIZE>{prompt}, destination);
    //ov_tokenizer.infer();
    ov_tokenizer.set_input_tensor(ov::Tensor{ ov::element::string, {BATCH_SIZE}, &prompt });
    ov_tokenizer.infer();
    return {ov_tokenizer.get_tensor("input_ids"), ov_tokenizer.get_tensor("attention_mask") };
}

std::string ChatGLM3Tokenizer::detokenize(std::vector<int64_t>& tokens) {
    constexpr size_t BATCH_SIZE = 1;
    ov_detokenizer.set_input_tensor(ov::Tensor{ ov::element::i64, {BATCH_SIZE, tokens.size()}, tokens.data() });
    ov_detokenizer.infer();
    return ov_detokenizer.get_output_tensor().data<std::string>()[0];
}

std::pair<ov::Tensor, ov::Tensor> ChatGLM3Tokenizer::encode(const std::string &text, int max_length) const {
    
    //sp.Encode(text, &ids);
    //std::string result = std::string("[gMASK]") + "sop" + text; // adding special prefix
    std::string result = text;
    
    //ids = tokenizer.get_tensor("input_ids");
    //attention_mask = tokenizer.get_tensor("attention_mask");
    //ids.insert(ids.begin(), {gmask_token_id, sop_token_id}); // special prefix
    
    truncate(result, max_length);   //TODO adding history clearn feature later.
    //auto [ids, attention_mask] = tokenize(result.c_str());
    tokenize(result.c_str());
    auto ids = ov_tokenizer.get_tensor("input_ids");
    auto attention_mask = ov_tokenizer.get_tensor("attention_mask");
    return {ids, attention_mask};
}

std::string ChatGLM3Tokenizer::decode(const std::vector<int> &ids) const {
    std::string text = decode_with_special_tokens(ids);
    text = remove_special_tokens(text);
    return text;
}

std::string ChatGLM3Tokenizer::decode_with_special_tokens(const std::vector<int> &ids) const {
    //std::vector<std::string> pieces;
    std::string text;
    constexpr size_t BATCH_SIZE = 1;
    //ov::Tensor inp = ov_detokenizer.get_input_tensor();
    //inp.set_shape({ BATCH_SIZE, 1 });
    std::vector<int64_t> ids_int64;
    std::transform(ids.begin(), ids.end(), std::back_inserter(ids_int64), [](int i) { return static_cast<int64_t>(i); });
    for (int id : ids) {
        auto pos = index_special_tokens.find(id);
        if (pos != index_special_tokens.end()) {
            // special tokens
            //pieces.emplace_back(pos->second);
            text += pos->second;
        } else {
            // normal tokens
            //inp.data<int64_t>()[0] = id;            
            ov_detokenizer.set_input_tensor(ov::Tensor{ ov::element::i64, {BATCH_SIZE, ids_int64.size()}, ids_int64.data() });
            ov_detokenizer.infer();
            //text += openvino_extensions::unpack_strings(ov_detokenizer.get_output_tensor()).front();
            text = ov_detokenizer.get_output_tensor().data<std::string>()[0];
            //pieces.emplace_back(sp.IdToPiece(id));
        }
    }
    //std::string text = sp.DecodePieces(pieces);    
    return text;
}

std::string ChatGLM3Tokenizer::remove_special_tokens(const std::string &text) {
    std::string output = text;
    static const std::vector<std::regex> special_token_regex{
        // std::regex(R"(<\|assistant\|> interpreter)"),
        // std::regex(R"(<\|assistant\|> interpre)"),
        std::regex(R"(<\|assistant\|>)"),
        std::regex(R"(<\|user\|>)"),
        std::regex(R"(<\|observation\|>)"),
    };
    for (const auto &re : special_token_regex) {
        //output = std::regex_replace(output, re, ""); //临时取消
    }
    return output;
}

std::string ChatGLM3Tokenizer::encode_single_message(const std::string &role, const std::string &content) const {
    //const ov::Tensor input_ids;
    std::string result = " <|" + role + "|>" + "\n" + content; // adding special token


    //input_ids.emplace_back(get_command("<|" + role + "|>"));
    // TODO: support metadata
    //const ov::Tensor newline_ids;
    //sp.Encode("\n", &newline_ids);
   // auto [newline_ids, attention_mask] = tokenize(tokenizer, "\n");

   // input_ids.insert(input_ids.end(), newline_ids.begin(), newline_ids.end());
    //const ov::Tensor content_ids;
    //sp.Encode(content, &content_ids);
    //auto [input_ids, attention_mask] = tokenize(tokenizer, result);
    //content_ids = tokenizer.get_tensor("input_ids");
    //attention_mask = tokenizer.get_tensor("attention_mask");
    //input_ids.insert(input_ids.end(), content_ids.begin(), content_ids.end());
    return result;
}

std::pair<ov::Tensor, ov::Tensor> ChatGLM3Tokenizer::encode_messages(const std::vector<ChatMessage> &messages, int max_length) const {

    //const ov::Tensor input_ids{gmask_token_id, sop_token_id};
    std::string result;
    //std::string result_perfix = std::string("[gMASK]") + "sop";
    for (const auto &msg : messages) {
        //result = result_perfix + encode_single_message(msg.role, msg.content);        
        result = "<|user|> " + msg.content + " <|assitant|>";
        //input_ids.insert(input_ids.end(), msg_ids.begin(), msg_ids.end());
        // encode code block into a separate message
        if (!msg.tool_calls.empty() && msg.tool_calls.front().type == ToolCallMessage::TYPE_CODE) {
            //result = result_perfix + encode_single_message(msg.role, msg.tool_calls.front().code.input);
            result = encode_single_message(msg.role, msg.tool_calls.front().code.input);
            //auto [code_ids, attention_mask] = encode_single_message(msg.role, msg.tool_calls.front().code.input);
            //input_ids.insert(input_ids.end(), code_ids.begin(), code_ids.end());
        }      
    }
    //input_ids.emplace_back(assistant_token_id);
    //result = result + " <|assistant|>";
    truncate(result, max_length);
    //auto [input_ids, attention_mask] = tokenize(result.c_str());
    tokenize(result.c_str());
    auto input_ids = ov_tokenizer.get_tensor("input_ids");
    auto attention_mask = ov_tokenizer.get_tensor("attention_mask");
    std::cout << "=====input length " << input_ids.get_size() << " =====\n" << std::endl;
    return { input_ids, attention_mask };
}

ChatMessage ChatGLM3Tokenizer::decode_message(const std::vector<int> &ids) const {
    ChatMessage message;
    if (!ids.empty() && ids.back() == observation_token_id) {
        // insert an <|assistant|> token before content to match possible interpreter delimiter
        std::vector<int> full_ids{assistant_token_id};
        full_ids.insert(full_ids.end(), ids.begin(), ids.end());

        std::string output = decode_with_special_tokens(full_ids);
        const std::string ci_delim = "<|assistant|> interpreter";
        size_t ci_pos = output.find(ci_delim);
        if (ci_pos != std::string::npos) {
            // code interpreter
            std::string chat_output = output.substr(0, ci_pos);
            chat_output = remove_special_tokens(chat_output);
            trim(chat_output);
            std::string code_output = output.substr(ci_pos + ci_delim.size());
            code_output = remove_special_tokens(code_output);
            trim(code_output);
            message = ChatMessage(ChatMessage::ROLE_ASSISTANT, std::move(chat_output),
                                  {ToolCallMessage(CodeMessage(std::move(code_output)))});
        } else {
            // tool call
            output = remove_special_tokens(output);

            // parse tool name
            std::string tool_name = "PARSE_ERROR";
            size_t pos = output.find('\n');
            if (pos != std::string::npos) {
                // split tool name and args by 1st linebreak
                tool_name = output.substr(0, pos);
                trim(tool_name);
                output.erase(0, pos + 1);
            }

            // post process output
            trim(output);

            // extract args
            std::string tool_args = "PARSE_ERROR";
            static const std::regex args_regex(R"(```.*?\n(.*?)\n```)");
            std::smatch sm;
            if (std::regex_search(output, sm, args_regex)) {
                CHATGLM_CHECK(sm.size() == 2) << "unexpected regex match results";
                tool_args = sm[1];
            }

            message = ChatMessage(ChatMessage::ROLE_ASSISTANT, std::move(output),
                                  {ToolCallMessage(FunctionMessage(std::move(tool_name), std::move(tool_args)))});
        }
    } else {
        // conversation
        //message = BaseTokenizer::decode_message(ids);
        message = { ChatMessage::ROLE_ASSISTANT, decode(ids) };
        trim(message.content); // strip leading linebreak in conversation mode
    }
    return message;
}

int ChatGLM3Tokenizer::get_command(const std::string &token) const {
    auto pos = special_tokens.find(token);
    CHATGLM_CHECK(pos != special_tokens.end()) << token << " is not a special token";
    return pos->second;
}

bool ChatGLM3Tokenizer::is_special_id(int id) const { return index_special_tokens.count(id) > 0; }

void ChatGLM3Tokenizer::truncate(std::string result, int max_length) {
    std::string result_perfix = std::string("[gMASK]") + "sop";
    if (result.size() > max_length) {
        // sliding window: drop the least recent history while keeping the two special prefix tokens
        int num_drop = result.size()- max_length;
        std::string preservedPart = result.substr(0, result_perfix.size());
        result = preservedPart + result.substr(result_perfix.size() + num_drop);
        //ids.erase(ids.begin() + 2, ids.begin() + 2 + num_drop);
    }
}


// ===== pipeline =====

Pipeline::Pipeline(const std::string &model_path, const std::string ov_device) {
#ifdef _DEBUG
#define USER_OV_EXTENSIONS_PATH "thirdparty/openvino_extension/lib_dll/debug_dll/user_ov_extensions.dll"
#else
#define USER_OV_EXTENSIONS_PATH "user_ov_extensions.dll"
#endif
    const std::string openvino_model_path = model_path + "modified_openvino_model.xml";
    const std::string tokenizer_path = model_path + "openvino_tokenizer.xml";
    const std::string detokenizer_path = model_path + "openvino_detokenizer.xml";
    std::cout << ov::get_openvino_version() << std::endl;
    core.add_extension(USER_OV_EXTENSIONS_PATH);  // USER_OV_EXTENSIONS_PATH is defined in root CMakeLists.txt
    if (std::filesystem::exists(tokenizer_path)) {
        // 文件存在，继续处理
    }
    else {
        // 文件不存在，处理错误
        std::cerr << "Error: Unable to read the model. File not found: " << tokenizer_path << std::endl;
    }
    chatglm3_tokenizer = std::make_unique<ChatGLM3Tokenizer>(core, tokenizer_path, detokenizer_path);

    //config.extra_eos_token_ids = { chatglm3_tokenizer->observation_token_id, chatglm3_tokenizer->user_token_id };
    
    

    device = ov_device;
    constexpr size_t BATCH_SIZE = 1;
    size_t convert_model = 0;// assume the model was already converted;

    ov::AnyMap device_config = {};
    if (device.find("CPU") != std::string::npos) {
        device_config[ov::cache_dir.name()] = "..\\..\\..\\..\\..\\model\\llm-cache";
        device_config[ov::hint::scheduling_core_type.name()] = ov::hint::SchedulingCoreType::PCORE_ONLY;
        device_config[ov::hint::enable_hyper_threading.name()] = false;
        device_config[ov::hint::enable_cpu_pinning.name()] = true;
        device_config[ov::enable_profiling.name()] = false;
    }

    if (device.find("GPU") != std::string::npos) {
        device_config[ov::cache_dir.name()] = "..\\..\\..\\..\\..\\model\\llm-cache";
        device_config[ov::intel_gpu::hint::queue_throttle.name()] = ov::intel_gpu::hint::ThrottleLevel::MEDIUM;
        device_config[ov::intel_gpu::hint::queue_priority.name()] = ov::hint::Priority::MEDIUM;
        device_config[ov::intel_gpu::hint::host_task_priority.name()] = ov::hint::Priority::HIGH;
        device_config[ov::hint::enable_cpu_pinning.name()] = true;
        device_config[ov::enable_profiling.name()] = false;
    }
    if (1 == convert_model) {
        auto startTime = Time::now();
        std::shared_ptr<ov::Model> model = core.read_model(openvino_model_path);
        auto duration_ms = get_duration_ms_until_now(startTime);
        std::cout << "=====Read chatglm Model took " << duration_ms << " ms=====\n" << std::endl;

        std::cout << "######## [Model Graph Optimization] Step 2: Insert slice node after reshape to reduce logits operation ########\n";
        ov::pass::Manager manager;
        manager.register_pass<InsertSlice>();
        manager.run_passes(model);

        std::string modifiled_file = std::regex_replace(openvino_model_path, std::regex("openvino_model"), "modified_openvino_model");
        std::cout << "Save modified model in " << modifiled_file << "\n";
        ov::serialize(model, modifiled_file);
        ov::CompiledModel compilemodel = core.compile_model(modifiled_file, device, device_config);

        throw std::invalid_argument("model covernt completed. please quite the application and restart");;
    }
    //Compile model
    model = std::make_unique<BaseModelForCausalLM>(core,openvino_model_path, device, device_config);




    //mapped_file = std::make_unique<MappedFile>(path);
    //ModelLoader loader(mapped_file->data, mapped_file->size);

    //// load magic
    //std::string magic = loader.read_string(4);
    //CHATGLM_CHECK(magic == "ggml") << "model file is broken (bad magic)";

    //// load model type
    //ModelType model_type = (ModelType)loader.read_basic<int>();
    //// load version
    //int version = loader.read_basic<int>();
    //if (model_type == ModelType::CHATGLM) {
    //    CHATGLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

    //    // load config
    //    ModelConfig config(model_type, loader.read_basic<ConfigRecordV1>());

    //    // load tokenizer
    //    int proto_size = loader.read_basic<int>();
    //    std::string_view serialized_model_proto((char *)mapped_file->data + loader.tell(), proto_size);
    //    loader.seek(proto_size, SEEK_CUR);
    //    tokenizer = std::make_unique<ChatGLMTokenizer>(serialized_model_proto);

    //    // load model
    //    model = std::make_unique<ChatGLMForCausalLM>(config);
    //    model->load(loader);
    //} else if (model_type == ModelType::CHATGLM2 || model_type == ModelType::CHATGLM3) {
    //    CHATGLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

    //    // load config
    //    ModelConfig config(model_type, loader.read_basic<ConfigRecordV2>());

    //    // load tokenizer
    //    int proto_size = loader.read_basic<int>();
    //    std::string_view serialized_model_proto((char *)mapped_file->data + loader.tell(), proto_size);
    //    loader.seek(proto_size, SEEK_CUR);

    //    if (model_type == ModelType::CHATGLM2) {
    //        tokenizer = std::make_unique<ChatGLM2Tokenizer>(serialized_model_proto);
    //        model = std::make_unique<ChatGLM2ForCausalLM>(config);
    //    } else {
    //        auto chatglm3_tokenizer = std::make_unique<ChatGLM3Tokenizer>(serialized_model_proto);
    //        config.extra_eos_token_ids = {chatglm3_tokenizer->observation_token_id, chatglm3_tokenizer->user_token_id};
    //        tokenizer = std::move(chatglm3_tokenizer);
    //        model = std::make_unique<ChatGLM3ForCausalLM>(config);
    //    }

    //    // load model
    //    model->load(loader);
    //} else if (model_type == ModelType::BAICHUAN7B) {
    //    CHATGLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

    //    // load config
    //    ModelConfig config(model_type, loader.read_basic<ConfigRecordV1>());
    //    config.norm_eps = 1e-6;

    //    // load tokenizer
    //    int proto_size = loader.read_basic<int>();
    //    std::string_view serialized_model_proto((char *)mapped_file->data + loader.tell(), proto_size);
    //    loader.seek(proto_size, SEEK_CUR);
    //    tokenizer = std::make_unique<BaichuanTokenizer>(serialized_model_proto);

    //    // load model
    //    model = std::make_unique<Baichuan7BForCausalLM>(config);
    //    model->load(loader);
    //} else if (model_type == ModelType::BAICHUAN13B) {
    //    CHATGLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

    //    // load config
    //    ModelConfig config(model_type, loader.read_basic<ConfigRecordV1>());
    //    config.norm_eps = 1e-6;

    //    // load tokenizer
    //    int proto_size = loader.read_basic<int>();
    //    std::string_view serialized_model_proto((char *)mapped_file->data + loader.tell(), proto_size);
    //    loader.seek(proto_size, SEEK_CUR);
    //    tokenizer = std::make_unique<BaichuanTokenizer>(serialized_model_proto);

    //    // load model
    //    model = std::make_unique<Baichuan13BForCausalLM>(config);
    //    model->load(loader);
    //} else if (model_type == ModelType::INTERNLM) {
    //    CHATGLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

    //    // load config
    //    ModelConfig config(model_type, loader.read_basic<ConfigRecordV1>());
    //    config.norm_eps = 1e-6;

    //    // load tokenizer
    //    int proto_size = loader.read_basic<int>();
    //    std::string_view serialized_model_proto((char *)mapped_file->data + loader.tell(), proto_size);
    //    loader.seek(proto_size, SEEK_CUR);
    //    tokenizer = std::make_unique<InternLMTokenizer>(serialized_model_proto);

    //    // load model
    //    if (config.hidden_size == 4096) {
    //        model = std::make_unique<InternLM7BForCausalLM>(config);
    //    } else {
    //        model = std::make_unique<InternLM20BForCausalLM>(config);
    //    }
    //    model->load(loader);
    //} else {
    //    CHATGLM_THROW << "invalid model type " << (int)model_type;
    //}
    
}

std::vector<int> Pipeline::generate(const ov::Tensor input_ids, const ov::Tensor attention_mask, const GenerationConfig &gen_config,
                                    BaseStreamer *streamer) const {
    std::vector<int> output_ids = model->generate(input_ids, attention_mask, gen_config, streamer);
    std::vector<int> new_output_ids(output_ids.begin() + input_ids.get_size(), output_ids.end());
    return new_output_ids;
}

std::string Pipeline::generate(const std::string &prompt, const GenerationConfig &gen_config,
                               BaseStreamer *streamer) const {
    auto [input_ids, attention_mask] = chatglm3_tokenizer->encode(prompt, gen_config.max_context_length);
    std::vector<int> new_output_ids = generate (input_ids, attention_mask, gen_config, streamer);
    std::string output = chatglm3_tokenizer->decode(new_output_ids);
    return output;
}

ChatMessage Pipeline::chat(const std::vector<ChatMessage> &messages, const GenerationConfig &gen_config,
                           BaseStreamer *streamer) const {

    auto [input_ids, attention_mask] = chatglm3_tokenizer->encode_messages(messages, gen_config.max_context_length);    
    std::vector<int> new_output_ids = generate (input_ids, attention_mask, gen_config, streamer);    
    ChatMessage output = chatglm3_tokenizer->decode_message(new_output_ids);
    //std::cout << "prompt_tokens: " << input_ids.get_size() << "\n";
    //std::cout << "completion_tokens: " << new_output_ids.size() << "\n";
    //std::cout << "output: " << output.content << "\n";
    return output;
}
ChatMessage Pipeline::chat_pygenerate(const std::vector<ChatMessage>& messages, const GenerationConfig& gen_config) {
    for (const auto& msg : messages)
        std::cout << "received message: "<<msg.content << std::endl;
    auto [input_ids, attention_mask] = chatglm3_tokenizer->encode_messages(messages, gen_config.max_context_length);
    prompt_tokens = input_ids.get_size();
    std::vector<int> new_output_ids = generate(input_ids, attention_mask, gen_config, nullptr);
    completion_tokens = new_output_ids.size();
    ChatMessage output = chatglm3_tokenizer->decode_message(new_output_ids);
    std::cout << "prompt_tokens: " << prompt_tokens<< "\n";
    std::cout << "completion_tokens: " << completion_tokens << "\n";
    std::cout << "output: " << output.content << "\n";
    return output;
}
void Pipeline::chat_pystream(const std::vector<ChatMessage>& messages, const GenerationConfig& gen_config, std::function<void(std::string, std::vector<int>)> &cb, std::string userdata) {
    // 创建文本流和性能统计流
    auto text_streamer = std::make_shared<chatglm::TextStreamer>(std::cout, chatglm3_tokenizer.get());
    text_streamer->py_streamer = true;
    text_streamer->py_userData = userdata;
    text_streamer->registerCallBack(cb);

    auto perf_streamer = std::make_shared<chatglm::PerfStreamer>();
    // 创建流处理器组，将文本流和性能统计流组合在一起
    std::vector<std::shared_ptr<chatglm::BaseStreamer>> streamers{ perf_streamer };
    streamers.emplace_back(text_streamer);
    auto streamer = std::make_unique<chatglm::StreamerGroup>(std::move(streamers));
        
    auto [input_ids, attention_mask] = chatglm3_tokenizer->encode_messages(messages, gen_config.max_context_length);
    prompt_tokens = input_ids.get_size();
    std::vector<int> new_output_ids = generate(input_ids, attention_mask, gen_config, streamer.get());
    completion_tokens = new_output_ids.size();
    ChatMessage output = chatglm3_tokenizer->decode_message(new_output_ids);
    std::cout << "prompt_tokens: " << prompt_tokens << "\n";
    std::cout << "completion_tokens: " << completion_tokens << "\n";
    std::cout << "output: " << output.content << "\n";
}

} // namespace chatglm
