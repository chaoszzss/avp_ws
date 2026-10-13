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

      // 初始化图像缓存
      current_images_.resize(4);

      // 初始化保存路径
      save_dir_ = "/home/chaoszzss/avp_ws/src/ipm_simulator/IPM_Dataset";
      raw_subdir_ = save_dir_ + "/raw";
      // 创建目录（如果不存在）
      if (!boost::filesystem::exists(raw_subdir_)) {
        boost::filesystem::create_directories(raw_subdir_);
      }
      last_save_time_ = ros::Time::now();
    }

    // ~IPMNode() {
    //     if (display_thread_.joinable()) {
    //         display_thread_.join();  // 等待显示线程结束
    //     }
    // }

    void Run() {
      // 使用 message_filters 同步订阅四个相机话题，只处理原始图像
      message_filters::Subscriber<sensor_msgs::CompressedImage> sub_front(nh_, "/camera/front", 1);
      message_filters::Subscriber<sensor_msgs::CompressedImage> sub_back(nh_, "/camera/back", 1);
      message_filters::Subscriber<sensor_msgs::CompressedImage> sub_left(nh_, "/camera/left", 1);
      message_filters::Subscriber<sensor_msgs::CompressedImage> sub_right(nh_, "/camera/right", 1);

      // 定义同步策略（4个话题）
      typedef message_filters::sync_policies::ApproximateTime<
          sensor_msgs::CompressedImage, sensor_msgs::CompressedImage,
          sensor_msgs::CompressedImage, sensor_msgs::CompressedImage> SyncPolicy;

      message_filters::Synchronizer<SyncPolicy> sync(
          SyncPolicy(10), sub_front, sub_back, sub_left, sub_right);
      sync.registerCallback(boost::bind(&IPMNode::imageCallback, this, _1, _2, _3, _4));

      // 仅发布IPM原始图像
      pub_ = it_.advertise("/ipm_raw", 1, true);

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
    double save_interval_;          // 保存间隔（秒）
    ros::Time last_save_time_;      // 上次保存时间
    double fps_{0.0};
    uint64_t frame_count_{0};
    int warmup_frames_{20};

    // 同步回调函数（接收四个相机的原始图像）
    void imageCallback(
        const sensor_msgs::CompressedImageConstPtr& front_msg,
        const sensor_msgs::CompressedImageConstPtr& back_msg,
        const sensor_msgs::CompressedImageConstPtr& left_msg,
        const sensor_msgs::CompressedImageConstPtr& right_msg) {

        // 定义互斥锁，保护图像
        std::lock_guard<std::mutex> lock(mutex_);
        
        try
        {
          // 将ROS压缩图像消息转换为OpenCV格式
          current_images_[0] = cv_bridge::toCvCopy(front_msg, "bgr8")->image; // front -> 0
          current_images_[1] = cv_bridge::toCvCopy(back_msg, "bgr8")->image;  // back -> 1
          current_images_[2] = cv_bridge::toCvCopy(left_msg, "bgr8")->image;  // left -> 2
          current_images_[3] = cv_bridge::toCvCopy(right_msg, "bgr8")->image; // right -> 3

          // 生成IPM图像
          auto start = std::chrono::high_resolution_clock::now();
          cv::Mat ipm_image = ipm_.GenerateIPMImage(current_images_);
          auto end = std::chrono::high_resolution_clock::now();
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

          // 如需保存IPM图像，可在此处按需写入raw_subdir_目录
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

          // 发布IPM原始图像
          sensor_msgs::ImagePtr msg = cv_bridge::CvImage(front_msg->header, "bgr8", ipm_image).toImageMsg();
          pub_.publish(msg);

          // 显示IPM原始图像
          showRawAndIPM(ipm_image);
          // 将显示的任务放到单独的线程中
        //   display_thread_ = std::thread(&IPMNode::showRawAndIPM, this, ipm_image);

        }
        
        catch(cv_bridge::Exception& e)
        {
          ROS_ERROR("CV Bridge error: %s", e.what());
        }
    }

    // 只显示IPM原始图像
    void showRawAndIPM(const cv::Mat& ipm_image) {
      cv::Mat ipm_display = ipm_image.clone();
      if (!ipm_display.empty()) {
        // std::string text = "Frame: " + std::to_string(frame_count_);
        std::string text;
        if (fps_ > 0.0) {
          text += " | FPS: " + std::to_string(static_cast<int>(std::round(fps_)));
        }
        cv::putText(ipm_display, text, cv::Point(10, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
      }
      cv::imshow("IPM Video", ipm_display);
      cv::waitKey(1);
    }

};


int main(int argc, char *argv[]) {
  // 初始化ROS节点
  setlocale(LC_ALL, "zh_CN.UTF-8");
  ros::init(argc, argv, "ipm_node_raw");

  // 创建IPM节点实例
  IPMNode node;
  node.Run();

  // 在循环结束后，销毁IPM图像窗口
  // cv::destroyWindow("IPM Video");
  // cv::destroyWindow("Raw Video");
  // cv::destroyWindow("Label Video");
  return 0;

}
