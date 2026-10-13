#include "ipm_fun.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

struct ImageSpec {
  int width{0};
  int height{0};
};

ImageSpec LoadImageSpec(const std::string& intrinsic_file) {
  ImageSpec spec;
  cv::FileStorage fs(intrinsic_file, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::cerr << "Failed to open intrinsic file: " << intrinsic_file << std::endl;
    return spec;
  }
  fs["image_width"] >> spec.width;
  fs["image_height"] >> spec.height;
  return spec;
}

void StampLabel(cv::Mat& image, const std::string& label) {
  cv::putText(image, label, cv::Point(40, 80), cv::FONT_HERSHEY_SIMPLEX,
              2.0, cv::Scalar(255, 255, 255), 6, cv::LINE_AA);
  cv::putText(image, label, cv::Point(40, 80), cv::FONT_HERSHEY_SIMPLEX,
              2.0, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
}

BlendingStrategy ParseStrategy(const std::string& name) {
  std::string lower = name;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  if (lower == "simple" || lower == "simple_average" || lower == "avg") {
    return BlendingStrategy::SIMPLE_AVERAGE;
  }
  if (lower == "weighted_simple" || lower == "simple_weighted") {
    return BlendingStrategy::WEIGHTED_SIMPLE;
  }
  if (lower == "hybrid") {
    return BlendingStrategy::HYBRID;
  }
  return BlendingStrategy::WEIGHTED_ADVANCED;
}

std::string StrategyToString(BlendingStrategy strategy) {
  switch (strategy) {
    case BlendingStrategy::SIMPLE_AVERAGE:
      return "SIMPLE_AVERAGE";
    case BlendingStrategy::WEIGHTED_SIMPLE:
      return "WEIGHTED_SIMPLE";
    case BlendingStrategy::WEIGHTED_ADVANCED:
      return "WEIGHTED_ADVANCED";
    case BlendingStrategy::HYBRID:
      return "HYBRID";
  }
  return "UNKNOWN";
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << "IPM weight export utility starting..." << std::endl;
  std::string data_dir = "image_data";
  std::string output_dir = "output";
  std::string tag = "synthetic";
  std::string strategy_name = "advanced";

  if (argc > 1) {
    data_dir = argv[1];
  }
  if (argc > 2) {
    output_dir = argv[2];
  }
  if (argc > 3) {
    tag = argv[3];
  }
  if (argc > 4) {
    strategy_name = argv[4];
  }

  std::vector<ImageSpec> specs;
  std::vector<std::string> intrinsic_files;
  std::vector<std::string> extrinsic_files;
  for (int idx = 0; idx < 4; ++idx) {
    const std::string suffix = std::to_string(idx);
    const std::string intrinsic = data_dir + "/" + suffix + "_intrinsic.yaml";
    const std::string extrinsic = data_dir + "/" + suffix + "_extrinsic.yaml";
    std::cout << "Loading camera " << idx << " intrinsics: " << intrinsic
              << " extrinsics: " << extrinsic << std::endl;
    intrinsic_files.emplace_back(intrinsic);
    extrinsic_files.emplace_back(extrinsic);
    ImageSpec spec = LoadImageSpec(intrinsic);
    if (spec.width <= 0 || spec.height <= 0) {
      std::cerr << "Invalid image size in " << intrinsic << std::endl;
      return 1;
    }
    specs.push_back(spec);
  }

  IPM ipm;
  for (int idx = 0; idx < 4; ++idx) {
    ipm.AddCamera(intrinsic_files[idx], extrinsic_files[idx]);
  }
  const BlendingStrategy strategy = ParseStrategy(strategy_name);
  ipm.SetBlendingStrategy(strategy);
  std::cout << "Using blending strategy: " << StrategyToString(strategy)
            << std::endl;
  std::cout << "Cameras added. Preparing synthetic inputs..." << std::endl;
  ipm.SetDebugWeightExport(true);

  std::vector<cv::Mat> images(4);
  const std::vector<cv::Scalar> palette = {
      cv::Scalar(255, 0, 0),    // Front - Blue
      cv::Scalar(0, 255, 0),    // Rear - Green
      cv::Scalar(0, 0, 255),    // Left - Red
      cv::Scalar(0, 255, 255)   // Right - Yellow
  };
  const std::vector<std::string> labels = {"Front", "Rear", "Left", "Right"};

  for (int idx = 0; idx < 4; ++idx) {
    const ImageSpec& spec = specs[idx];
    images[idx] = cv::Mat(spec.height, spec.width, CV_8UC3, palette[idx]);
    StampLabel(images[idx], labels[idx]);
  }

  cv::utils::fs::createDirectories(output_dir);

  std::cout << "Generating IPM image..." << std::endl;

  cv::Mat ipm_image = ipm.GenerateIPMImage(images);
  const std::string ipm_path =
      output_dir + "/ipm_debug_" + tag + ".png";
  if (!cv::imwrite(ipm_path, ipm_image)) {
    std::cerr << "Failed to write IPM image to " << ipm_path << std::endl;
  } else {
    std::cout << "IPM image written to " << ipm_path << std::endl;
  }

  std::cout << "Exporting weight maps..." << std::endl;
  ipm.ExportWeightDebugImages(output_dir, tag);

  std::cout << "Weight debug artifacts exported to " << output_dir
            << " (tag: " << tag << ")" << std::endl;
  return 0;
}
