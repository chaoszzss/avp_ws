#pragma once
#include "avp_mapping/map.h"

class ViewerInterface {
 public:
  // display map
  virtual void displayAvpMap(const TimedPose &pose, const Map &avp_map, const std::vector<Slot> &cur_slots)= 0;
  // display ipm detection result
  void displayIpmDetection(const std::vector<cv::Point2f> &corners,
                           const std::vector<cv::Point2f> &slots_corners,
                           const cv::Mat &gray_image,
                           const std::vector<bool> &occupancy_status,
                           const std::vector<float> &slot_energies = {},
                           const std::vector<std::array<float, 3>> &slot_visibilities = {}) {  // 添加占用状态参数
    // std::cout << "\tipm detect : " << corners.size() << " corners, " << slots_corners.size() / 4 << " slots" << std::endl;
    if (imshow_) {
      cv::imshow("ipm_detection_result", drawIPMImg(corners, slots_corners, gray_image,
                                                   occupancy_status, slot_energies,
                                                   slot_visibilities));
      int key = cv::waitKey(delay_);
      if ('r' == key) {
        delay_ = 1;  // set auto run
      }
      if ('s' == key) {
        delay_ = 0;  // run image by image
      }
    }
  }

 private:
  cv::Mat drawIPMImg(const std::vector<cv::Point2f> &corners,
                     const std::vector<cv::Point2f> &slots_corners,
                     const cv::Mat &gray_image,
                     const std::vector<bool> &occupancy_status,
                     const std::vector<float> &slot_energies,
                     const std::vector<std::array<float, 3>> &slot_visibilities) {  // 添加占用状态参数
    // 将灰度图像转换为彩色图像以便绘制彩色标记
    cv::Mat color_image;
    cv::cvtColor(gray_image, color_image, cv::COLOR_GRAY2BGR);
    
    for (const auto &center : corners) {
      cv::circle(color_image, center, 5, cv::Scalar(0, 255, 255), 2);  // 黄色圆圈标记角点
    }
    
    for (int i = 0; i < slots_corners.size(); i += 4) {
      const int slot_index = i / 4;
      cv::Scalar color = occupancy_status[slot_index] ? cv::Scalar(0, 0, 255)
                                                      : cv::Scalar(0, 255, 0);  // 红色表示占用，绿色表示空闲
      cv::line(color_image, slots_corners[i], slots_corners[i + 1], color, 2);
      cv::line(color_image, slots_corners[i + 1], slots_corners[i + 2], color, 2);
      cv::line(color_image, slots_corners[i + 2], slots_corners[i + 3], color, 2);
      cv::line(color_image, slots_corners[i + 3], slots_corners[i], color, 2);
      
      // 在车位中心添加占用标识
      cv::Point2f center = (slots_corners[i] + slots_corners[i + 1] + 
                           slots_corners[i + 2] + slots_corners[i + 3]) / 4.0f;
      if (occupancy_status[slot_index]) {
        cv::putText(color_image, "X", center, cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
      } else {
        cv::putText(color_image, "O", center, cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
      }

      if (slot_index < static_cast<int>(slot_energies.size())) {
        const float energy = slot_energies[slot_index];
        const auto &vis = slot_visibilities.size() > slot_index
                              ? slot_visibilities[slot_index]
                              : std::array<float, 3>{{0.f, 0.f, 0.f}};
        const std::string info =
            cv::format("E:%.1f v:[%.2f %.2f %.2f]", energy, vis[0], vis[1], vis[2]);
        cv::putText(color_image, info, center + cv::Point2f(-40.f, 20.f),
                    cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 255, 0), 1);
      }
    }
    return color_image;
  }

 protected:
  const bool imshow_; // do cv imshow
  int delay_;
  ViewerInterface(bool imshow, bool autorun) : imshow_(imshow), delay_(0) {
    if (imshow_) {
      cv::namedWindow("ipm_detection_result", cv::WINDOW_NORMAL);
    }
    if (autorun){
      delay_ = 1;
    }
  }
  ~ViewerInterface() {
    cv::destroyAllWindows();
  }
};
