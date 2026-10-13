#include "ipm_fun.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <dirent.h>
#include <mutex>
#include <thread>
#include <climits>
#include <boost/filesystem.hpp>

// ROS
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#define IMAGE_DIR "/home/chaoszzss/avp_ws/src/ipm_simulator/image_data/"

class IPMNode {
public:
    IPMNode() : it_(nh_), save_interval_(0.05)
    {
      Eigen::initParallel();
      
      // 添加相机参数
      for (int idx = 0; idx < 4; ++idx) {
        std::string ex_file = std::string(IMAGE_DIR) + std::to_string(idx) + "_extrinsic.yaml";
        std::string in_file = std::string(IMAGE_DIR) + std::to_string(idx) + "_intrinsic.yaml";
        ipm_.AddCamera(in_file, ex_file);
      }

      // 创建窗口
      cv::namedWindow("IPM Semantic Label", cv::WINDOW_NORMAL);
      cv::resizeWindow("IPM Semantic Label", 800, 600);

      // 初始化分割图像缓存
      current_seg_images_.resize(4);

      // 初始化保存路径
      save_dir_ = "/home/chaoszzss/avp_ws/src/ipm_simulator/IPM_Dataset";
      label_subdir_ = save_dir_ + "/label";

      // 创建目录
      if (!boost::filesystem::exists(label_subdir_)) {
          boost::filesystem::create_directories(label_subdir_);
      }
      last_save_time_ = ros::Time::now();
    }

    void Run() {
        // 只订阅四个分割相机话题
        message_filters::Subscriber<sensor_msgs::CompressedImage> sub_front_seg(nh_, "/camera/front_seg", 1);
        message_filters::Subscriber<sensor_msgs::CompressedImage> sub_back_seg(nh_, "/camera/back_seg", 1);
        message_filters::Subscriber<sensor_msgs::CompressedImage> sub_left_seg(nh_, "/camera/left_seg", 1);
        message_filters::Subscriber<sensor_msgs::CompressedImage> sub_right_seg(nh_, "/camera/right_seg", 1);
    
        // 定义同步策略（4个话题）
        typedef message_filters::sync_policies::ApproximateTime<
            sensor_msgs::CompressedImage, sensor_msgs::CompressedImage,
            sensor_msgs::CompressedImage, sensor_msgs::CompressedImage> SyncPolicy;
        
        message_filters::Synchronizer<SyncPolicy> sync(
          SyncPolicy(10), sub_front_seg, sub_back_seg, sub_left_seg, sub_right_seg);
        sync.registerCallback(boost::bind(&IPMNode::imageCallback, this, _1, _2, _3, _4));
    
        // 只发布IPM分割标签
        pub_seg_ = it_.advertise("/ipm_semantic_label", 1, true);
    
        ros::spin();
    }

private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    IPM ipm_;
    
    std::vector<cv::Mat> current_seg_images_;  // 分割图像缓存
    image_transport::Publisher pub_seg_;        // 分割IPM发布者
    std::mutex mutex_;

    std::string save_dir_;
    std::string label_subdir_;
    double save_interval_;
    ros::Time last_save_time_;

    double fps_;
    int warmup_frames_{20};
    
    void quantizeLabelImage(cv::Mat& image) {
      const std::vector<cv::Vec3b> color_palette = {
          cv::Vec3b(130, 63, 122),
          cv::Vec3b(0, 254, 254),
          cv::Vec3b(45, 192, 112),
          cv::Vec3b(0, 0, 0),
          cv::Vec3b(136, 133, 33),
          cv::Vec3b(254, 254, 254),
          cv::Vec3b(0, 191, 192),
          cv::Vec3b(254, 128, 76)
      };

      if (image.empty()) {
        return;
      }
      
      const int rows = image.rows;
      const int cols = image.cols;
  
      #pragma omp parallel for schedule(static, 32)
      for (int y = 0; y < rows; y++) {
          cv::Vec3b* row_ptr = image.ptr<cv::Vec3b>(y);
          for (int x = 0; x < cols; x++) {
              cv::Vec3b& pixel = row_ptr[x];
              int min_dist = INT_MAX;
              int best_idx = 0;
  
              // 精确匹配
              bool exact_match = false;
              for (size_t i = 0; i < color_palette.size(); i++) {
                  const cv::Vec3b& ref = color_palette[i];
                  if (std::abs(pixel[0] - ref[0]) <= 1 &&
                      std::abs(pixel[1] - ref[1]) <= 1 &&
                      std::abs(pixel[2] - ref[2]) <= 1) {
                      best_idx = i;
                      exact_match = true;
                      break;
                  }
              }
  
              // 最近邻匹配
              if (!exact_match) {
                  for (size_t i = 0; i < color_palette.size(); i++) {
                      const cv::Vec3b& ref = color_palette[i];
                      int dist = (pixel[0]-ref[0])*(pixel[0]-ref[0]) + 
                                 (pixel[1]-ref[1])*(pixel[1]-ref[1]) + 
                                 (pixel[2]-ref[2])*(pixel[2]-ref[2]);
                      if (dist < min_dist) {
                          min_dist = dist;
                          best_idx = i;
                      }
                  }
              }
  
              pixel = color_palette[best_idx];
          }
      }
    }

    void imageCallback(
                      const sensor_msgs::CompressedImageConstPtr& front_seg_msg,
                      const sensor_msgs::CompressedImageConstPtr& back_seg_msg,
                      const sensor_msgs::CompressedImageConstPtr& left_seg_msg,
                      const sensor_msgs::CompressedImageConstPtr& right_seg_msg) 
    {
        std::lock_guard<std::mutex> lock(mutex_);
        
        try
        {
          // 只处理分割图像
          current_seg_images_[0] = cv_bridge::toCvCopy(front_seg_msg, "bgr8")->image;
          current_seg_images_[1] = cv_bridge::toCvCopy(back_seg_msg, "bgr8")->image;
          current_seg_images_[2] = cv_bridge::toCvCopy(left_seg_msg, "bgr8")->image;
          current_seg_images_[3] = cv_bridge::toCvCopy(right_seg_msg, "bgr8")->image;

          auto start = std::chrono::high_resolution_clock::now();
          
          // 只生成分割IPM
          cv::Mat ipm_seg_image = ipm_.GenerateIPMImage(current_seg_images_, true);
          quantizeLabelImage(ipm_seg_image);
          
          auto end = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
          
          if (warmup_frames_ > 0) {
            --warmup_frames_;
          } else {
            ROS_INFO("IPM Semantic Label generation: %ld ms", duration.count());
          }

          if (duration.count() > 0) {
            fps_ = 1000.0 / duration.count();
          } else {
            fps_ = 0.0;
          }
          
          // 显示FPS
          std::string fps_text = "FPS: " + std::to_string(static_cast<int>(std::round(fps_)));
          cv::putText(ipm_seg_image, fps_text, cv::Point(10, 40),
                     cv::FONT_HERSHEY_SIMPLEX, 1.6, cv::Scalar(0, 255, 0), 2);

          // 发布和显示
          sensor_msgs::ImagePtr msg_seg = cv_bridge::CvImage(front_seg_msg->header, "bgr8", ipm_seg_image).toImageMsg();
          pub_seg_.publish(msg_seg);
          
          cv::imshow("IPM Semantic Label", ipm_seg_image);
          cv::waitKey(1);
        }
        catch(cv_bridge::Exception& e)
        {
          ROS_ERROR("CV Bridge error: %s", e.what());
        }
    }
};

int main(int argc, char *argv[]) {
  setlocale(LC_ALL, "zh_CN.UTF-8");
  ros::init(argc, argv, "ipm_node");

  IPMNode node;
  node.Run();

  return 0;
}
