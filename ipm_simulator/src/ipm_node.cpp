#include "ipm_fun.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <dirent.h>
#include <climits>
#include <cstdint>
#include <mutex>
#include <thread>
#include <boost/filesystem.hpp>

// ROS
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#define IMAGE_DIR "/home/chaoszzss/avp_ws/src/ipm_simulator/image_data/"  // 定义图像数据读取路径

class IPMNode {
public:
    IPMNode() : it_(nh_), save_interval_(0.05) /* 默认1秒保存一次 */
    {
      Eigen::initParallel();
      // 添加相机参数
      for (int idx = 0; idx < 4; ++idx) {
        std::string ex_file = std::string(IMAGE_DIR) + std::to_string(idx) + "_extrinsic.yaml";
        std::string in_file = std::string(IMAGE_DIR) + std::to_string(idx) + "_intrinsic.yaml";
        ipm_.AddCamera(in_file, ex_file);
      }

      // 创建窗口（仅一次）
      cv::namedWindow("IPM Video", cv::WINDOW_NORMAL);
      cv::resizeWindow("IPM Video", 800,600);
      cv::namedWindow("Label Video", cv::WINDOW_NORMAL);
      cv::resizeWindow("Label Video", 800,600);

      // 初始化图像缓存
      current_images_.resize(4);
      current_seg_images_.resize(4);

      // 初始化保存路径
      save_dir_ = "/home/chaoszzss/avp_ws/src/ipm_simulator/IPM_Dataset";
      raw_subdir_ = save_dir_ + "/raw";
      label_subdir_ = save_dir_ + "/label";

      // 创建目录（如果不存在）
      if (!boost::filesystem::exists(raw_subdir_)) {
        boost::filesystem::create_directories(raw_subdir_);
      }
      if (!boost::filesystem::exists(label_subdir_)) {
          boost::filesystem::create_directories(label_subdir_);
      }
      last_save_time_ = ros::Time::now();
    }

    // ~IPMNode() {
    //     if (display_thread_.joinable()) {
    //         display_thread_.join();  // 等待显示线程结束
    //     }
    // }

    void Run() {
        // 使用 message_filters 同步订阅四个相机话题
        message_filters::Subscriber<sensor_msgs::CompressedImage> sub_front(nh_, "/camera/front", 1);
        message_filters::Subscriber<sensor_msgs::CompressedImage> sub_back(nh_, "/camera/back", 1);
        message_filters::Subscriber<sensor_msgs::CompressedImage> sub_left(nh_, "/camera/left", 1);
        message_filters::Subscriber<sensor_msgs::CompressedImage> sub_right(nh_, "/camera/right", 1);

        // 订阅label话题
        message_filters::Subscriber<sensor_msgs::CompressedImage> sub_front_seg(nh_, "/camera/front_seg", 1);
        message_filters::Subscriber<sensor_msgs::CompressedImage> sub_back_seg(nh_, "/camera/back_seg", 1);
        message_filters::Subscriber<sensor_msgs::CompressedImage> sub_left_seg(nh_, "/camera/left_seg", 1);
        message_filters::Subscriber<sensor_msgs::CompressedImage> sub_right_seg(nh_, "/camera/right_seg", 1);
    
        // 定义同步策略（8个话题）
        typedef message_filters::sync_policies::ApproximateTime<
            sensor_msgs::CompressedImage, sensor_msgs::CompressedImage,
            sensor_msgs::CompressedImage, sensor_msgs::CompressedImage,
            sensor_msgs::CompressedImage, sensor_msgs::CompressedImage,
            sensor_msgs::CompressedImage, sensor_msgs::CompressedImage> SyncPolicy;
        
        message_filters::Synchronizer<SyncPolicy> sync(
          SyncPolicy(10), sub_front, sub_back, sub_left, sub_right,
          sub_front_seg, sub_back_seg, sub_left_seg, sub_right_seg);
          sync.registerCallback(boost::bind(&IPMNode::imageCallback, this, _1, _2, _3, _4, _5, _6, _7, _8));
    
        // 发布IPM图像的话题，true表示关闭压缩
        pub_ = it_.advertise("/ipm_raw", 1, true);
        pub_seg_ = it_.advertise("/ipm_label", 1, true);
    
        ros::spin();
      }

private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Publisher pub_;
    IPM ipm_;
    std::vector<cv::Mat> current_images_;  // 缓存当前图像
    std::mutex mutex_;  // 互斥锁，保护图像缓存
    std::thread display_thread_;  // 显示线程

    std::vector<cv::Mat> current_seg_images_;  // 分割图像缓存
    image_transport::Publisher pub_seg_;       // 分割IPM发布者

    std::string save_dir_;          // 图像保存路径
    std::string raw_subdir_;      // 原始图像保存路径
    std::string label_subdir_;    // 标签图像保存路径
    double save_interval_;          // 保存间隔（秒）
    ros::Time last_save_time_;      // 上次保存时间
    double fps_{0.0};
    uint64_t frame_count_{0};
    int warmup_frames_{20};
    
    // 定义一个函数，将分割图像的标签映射到精确的RGB颜色
    cv::Mat quantizeLabelImage(const cv::Mat& input) {
      // 修正颜色映射为BGR顺序，与输入图像格式一致
      const std::vector<cv::Vec3b> color_palette = {
          cv::Vec3b(130, 63, 122),    // 类别1 (原RGB:122,63,130 → BGR:130,63,122)
          cv::Vec3b(0, 254, 254),     // 类别2 (原RGB:254,254,0 → BGR:0,254,254)
          cv::Vec3b(45, 192, 112),    // 类别3 (原RGB:112,192,45 → BGR:45,192,112)
          cv::Vec3b(0, 0, 0),         // 类别4 (保持不变)
          cv::Vec3b(136, 133, 33),    // 类别5 (原RGB:33,133,136 → BGR:136,133,33)
          cv::Vec3b(254, 254, 254),   // 类别6 (保持不变)
          cv::Vec3b(0, 191, 192),     // 类别7 (原RGB:192,191,0 → BGR:0,191,192)
          cv::Vec3b(254, 128, 76)     // 类别8 (原RGB:76,128,254 → BGR:254,128,76)
      };
  
      cv::Mat output = input.clone();
      const int rows = input.rows;
      const int cols = input.cols;
  
      // 并行化遍历像素
      #pragma omp parallel for
      for (int y = 0; y < rows; y++) {
          for (int x = 0; x < cols; x++) {
              const cv::Vec3b& pixel = input.at<cv::Vec3b>(y, x);
              int min_dist = INT_MAX;
              int best_idx = 0;
  
              // 精确匹配：检查是否为预设颜色，允许±1的容差以应对压缩失真
              bool exact_match = false;
              for (size_t i = 0; i < color_palette.size(); i++) {
                  const cv::Vec3b& ref = color_palette[i];
                  // 允许每个通道有±1的误差
                  if (std::abs(pixel[0] - ref[0]) <= 1 &&
                      std::abs(pixel[1] - ref[1]) <= 1 &&
                      std::abs(pixel[2] - ref[2]) <= 1) {
                      best_idx = i;
                      exact_match = true;
                      break;
                  }
              }
  
              if (!exact_match) {
                  // 未精确匹配时，选择欧氏距离最小的颜色
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
  
              output.at<cv::Vec3b>(y, x) = color_palette[best_idx];
          }
      }
      return output;
  }

    // 同步回调函数（接收四个相机的图像）
    void imageCallback(
                      const sensor_msgs::CompressedImageConstPtr& front_msg,
                      const sensor_msgs::CompressedImageConstPtr& back_msg,
                      const sensor_msgs::CompressedImageConstPtr& left_msg,
                      const sensor_msgs::CompressedImageConstPtr& right_msg,
                      const sensor_msgs::CompressedImageConstPtr& front_seg_msg,
                      const sensor_msgs::CompressedImageConstPtr& back_seg_msg,
                      const sensor_msgs::CompressedImageConstPtr& left_seg_msg,
                      const sensor_msgs::CompressedImageConstPtr& right_seg_msg
                      ) 
    {

        // 定义互斥锁，保护图像
        std::lock_guard<std::mutex> lock(mutex_);
        
        try
        {
          // 将ROS图像消息转换为OpenCV格式
        //   cv::Mat front_image = cv_bridge::toCvCopy(front, "bgr8")->image;
          current_images_[0] = cv_bridge::toCvCopy(front_msg, "bgr8")->image; // front -> 0
          current_images_[1] = cv_bridge::toCvCopy(back_msg, "bgr8")->image;  // back -> 1
          current_images_[2] = cv_bridge::toCvCopy(left_msg, "bgr8")->image;  // left -> 2
          current_images_[3] = cv_bridge::toCvCopy(right_msg, "bgr8")->image; // right -> 3

          // 处理分割图像（假设分割图为单通道）
          current_seg_images_[0] = cv_bridge::toCvCopy(front_seg_msg, "bgr8")->image;
          current_seg_images_[1] = cv_bridge::toCvCopy(back_seg_msg, "bgr8")->image;
          current_seg_images_[2] = cv_bridge::toCvCopy(left_seg_msg, "bgr8")->image;
          current_seg_images_[3] = cv_bridge::toCvCopy(right_seg_msg, "bgr8")->image;

          // 生成IPM图像
          auto start = std::chrono::high_resolution_clock::now();
          cv::Mat ipm_image = ipm_.GenerateIPMImage(current_images_);
          auto end = std::chrono::high_resolution_clock::now();
          cv::Mat ipm_seg_image = ipm_.GenerateIPMImage(current_seg_images_);
          // 新增：对分割图进行颜色量化预处理
          ipm_seg_image = quantizeLabelImage(ipm_seg_image);
          auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
          if (warmup_frames_ > 0) {
            --warmup_frames_;
          } else {
            ROS_INFO("Processing time: %ld ms", duration.count());
          }
          if (duration.count() > 0) {
            fps_ = 1000.0 / static_cast<double>(duration.count());
          } else {
            fps_ = 0.0;
          }
          ++frame_count_;

          // 保存IPM图像
          // ros::Time current_time = ros::Time::now();
          // if ((current_time - last_save_time_).toSec() >= save_interval_) {
          //     // 生成时间戳文件名
          //     std::string timestamp = std::to_string(current_time.toNSec());
          //     std::string raw_path = raw_subdir_ + "/" + timestamp + ".png";
          //     std::string label_path = label_subdir_ + "/" + timestamp + ".png";
              
          //     // 保存图像
          //     cv::imwrite(raw_path, ipm_image);
          //     cv::imwrite(label_path, ipm_seg_image);
          //     ROS_INFO("Saved images: %s, %s", raw_path.c_str(), label_path.c_str());
              
          //     last_save_time_ = current_time; // 更新时间戳
          // }

          // 发布IPM图像
          sensor_msgs::ImagePtr msg = cv_bridge::CvImage(front_msg->header, "bgr8", ipm_image).toImageMsg();
          sensor_msgs::ImagePtr msg_seg = cv_bridge::CvImage(front_seg_msg->header, "bgr8", ipm_seg_image).toImageMsg();
          pub_.publish(msg);
          pub_seg_.publish(msg_seg);

          // 显示原始图像和IPM图像
          showRawAndIPM(ipm_image, ipm_seg_image);
          // 将显示的任务放到单独的线程中
        //   display_thread_ = std::thread(&IPMNode::showRawAndIPM, this, ipm_image);

        }
        
        catch(cv_bridge::Exception& e)
        {
          ROS_ERROR("CV Bridge error: %s", e.what());
        }
    }

    void showRawAndIPM(const cv::Mat& ipm_image, const cv::Mat& ipm_seg_image) {
      // 拼接原始图像
      // cv::Mat raw_front_back, raw_left_right, raw_display;
      // cv::hconcat(current_images_[1], current_images_[0], raw_front_back);   // 后+前
      // cv::hconcat(current_images_[2], current_images_[3], raw_left_right);   // 左+右
      // cv::vconcat(raw_front_back, raw_left_right, raw_display);

      // 显示
      cv::Mat ipm_display = ipm_image.clone();
      cv::Mat label_display = ipm_seg_image.clone();
      if (!ipm_display.empty()) {
        std::string text = "Frame: " + std::to_string(frame_count_);
        if (fps_ > 0.0) {
          text += " | FPS: " + std::to_string(static_cast<int>(std::round(fps_)));
        }
        cv::putText(ipm_display, text, cv::Point(10, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
      }
      if (!label_display.empty()) {
        std::string text = "Frame: " + std::to_string(frame_count_);
        cv::putText(label_display, text, cv::Point(10, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
      }
      cv::imshow("IPM Video", ipm_display);
      cv::imshow("Label Video", label_display);
      // cv::imshow("Raw Video", raw_display);
      cv::waitKey(1);
    }

};


int main(int argc, char *argv[]) {
  // 初始化ROS节点
  setlocale(LC_ALL, "zh_CN.UTF-8");
  ros::init(argc, argv, "ipm_node");

  // 创建IPM节点实例
  IPMNode node;
  node.Run();

  // 在循环结束后，销毁IPM图像窗口
  // cv::destroyWindow("IPM Video");
  // cv::destroyWindow("Raw Video");
  // cv::destroyWindow("Label Video");
  return 0;

}
