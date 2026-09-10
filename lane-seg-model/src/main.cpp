#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "lane_seg.hpp"

int main()
{
    LaneSegModel model("models/best.onnx");
    std::cout << "Model Loaded" << std::endl;

    std::string directory_path = "images/";

    std::vector<cv::String> image_files;
    cv::glob(directory_path + "*.jpg", image_files);

    for (int i = 0; i < image_files.size(); ++i)
    {
        cv::Mat img = cv::imread(image_files[i]);

        if (img.empty())
        {
            std::cerr << "Error: Could not read the image " << image_files[i] << std::endl;
            continue;
        }

        cv::resize(img, img, cv::Size(640, 360));
        cv::Mat img_vis = img.clone();

        cv::Mat da_out, ll_out;

        model.Infer(img, da_out, ll_out);
        std::cout << "Inference Done for " << image_files[i] << std::endl;

        img_vis.setTo(cv::Scalar(255, 0, 0), da_out);
        img_vis.setTo(cv::Scalar(0, 255, 255), ll_out);

        std::string result_path = "results" + std::to_string(i) + ".jpg";
        cv::imwrite(result_path, img_vis);
        std::cout << "Results saved to " << result_path << std::endl;
    }

    return 0;
}
