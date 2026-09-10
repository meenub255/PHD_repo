#include "lane_seg.hpp"

LaneSegModel::LaneSegModel(std::string model_path, int cuda_device_id)
{
    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "lane_seg");

#ifdef ENABLE_CUDA
    cuda_option_.device_id = cuda_device_id;
    session_options_.AppendExecutionProvider_CUDA(cuda_option_);
#endif

    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = new Ort::Session(env_, model_path.c_str(), session_options_);
    run_options_ = Ort::RunOptions{nullptr};

    size_t input_nodes_num = session_->GetInputCount();
    for (size_t i = 0; i < input_nodes_num; i++)
    {
        Ort::AllocatedStringPtr input_name = session_->GetInputNameAllocated(i, allocator_);
        char *temp_buf = new char[50];
        strcpy(temp_buf, input_name.get());
        input_node_names_.push_back(temp_buf);
    }

    size_t output_node_num = session_->GetOutputCount();
    for (size_t i = 0; i < output_node_num; i++)
    {
        Ort::AllocatedStringPtr output_node_name = session_->GetOutputNameAllocated(i, allocator_);
        char *temp_buf = new char[10];
        strcpy(temp_buf, output_node_name.get());
        output_node_names_.push_back(temp_buf);
    }
}

LaneSegModel::~LaneSegModel()
{
    delete session_;
}

void LaneSegModel::Infer(const cv::Mat &image, cv::Mat &da_out, cv::Mat &ll_out)
{
    assert(image.rows == input_node_dims_[2] && image.cols == input_node_dims_[3] && image.channels() == input_node_dims_[1]);

    cv::Mat input_img = image.clone();

    cv::Mat blob = cv::dnn::blobFromImage(input_img,
                                          1.0 / 255.0,
                                          cv::Size(input_node_dims_[3],
                                                   input_node_dims_[2]),
                                          cv::Scalar(), true, false);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
                                                              blob.ptr<float>(),
                                                              std::accumulate(input_node_dims_.begin(),
                                                                              input_node_dims_.end(),
                                                                              1, std::multiplies<int>()),
                                                              input_node_dims_.data(),
                                                              input_node_dims_.size());

    auto output_tensor = session_->Run(run_options_, input_node_names_.data(), &input_tensor,
                                       1, output_node_names_.data(),
                                       output_node_names_.size());

    std::vector<cv::Mat> results;

    for (int head_idx = 0; head_idx < output_node_names_.size(); head_idx++)
    {
        cv::Mat img_out(per_outsection_dims[1], per_outsection_dims[2], CV_32F);
        float *output_data = output_tensor[head_idx].GetTensorMutableData<float>();
        std::vector<float> output_vector(output_data,
                                         output_data + std::accumulate(per_outsection_dims.begin(),
                                                                       per_outsection_dims.end(),
                                                                       1, std::multiplies<int>()));
        std::memcpy(img_out.data,
                    output_vector.data() + per_outsection_dims[1] * per_outsection_dims[2],
                    per_outsection_dims[1] * per_outsection_dims[2] * sizeof(float));
        results.push_back(img_out.clone());
    }

    results[0].convertTo(da_out, CV_8UC1, 255.0);
    results[1].convertTo(ll_out, CV_8UC1, 255.0);
}
