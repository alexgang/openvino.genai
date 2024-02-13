@echo on
call "\Program Files (x86)\Intel\openvino_2023.3.0\setupvars.bat"
cd out\build\x64-Release\llm\chatglm_cpp
.\chatglm.exe "..\..\..\..\..\..\ChatGLM3_C++_demo\model\GPTQ_INT4-FP16\openvino_model.xml" "..\..\..\..\..\..\ChatGLM3_C++_demo\model\GPTQ_INT4-FP16\tokenizer.xml" "..\..\..\..\..\..\ChatGLM3_C++_demo\model\GPTQ_INT4-FP16\detokenizer.xml" "GPU" "hi"