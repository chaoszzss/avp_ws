#pragma once
#include <functional>
#include <mutex>
#include <thread>

#include "map.h"

// 地图可视化类
class MapViewer {
 public:
  // 析构函数
  ~MapViewer() {
    count_ = -1;
    thread_.join();
  }
  // 构造函数
  MapViewer(const Map &avp_map) {
    // 获取地图边界框
    bounding_box_ = avp_map.getBoundingBox();
    if (bounding_box_.isEmpty()) {  // empty map
      // 如果地图为空，则扩展边界框
      bounding_box_.extend(Eigen::Vector3d(0, 0, 0));
    }
    // 计算地图的宽度和高度
    Eigen::Vector3d bl = bounding_box_.min() + Eigen::Vector3d(-1, -1, -1);
    Eigen::Vector3d tr = bounding_box_.max() + Eigen::Vector3d(1, 1, 1);
    // 计算地图的像素宽度
    int width = (tr.x() - bl.x()) * kPixelScaleInv;
    // 计算地图的像素高度
    int height = (tr.y() - bl.y()) * kPixelScaleInv;
    // 创建地图图像
    map_ = cv::Mat(height, width, CV_8UC3, cv::Vec3b(91, 91, 91));
    // 计算地图的像素坐标
    to_map_pixel_ = [=](const Eigen::Vector3d &pt) {
      return cv::Point2f((pt.x() - bl.x()) * kPixelScaleInv,
                         height - (pt.y() - bl.y()) * kPixelScaleInv);
    };
    // 偏移量的像素坐标
    offset_ = to_map_pixel_(Eigen::Vector3d(0, 0, 0));
    // 绘制车位点
    drawGrid(map_, avp_map.getSemanticElement(SemanticLabel::kSlot),
             cv::Vec3b(kSlotGray, kSlotGray, kSlotGray));
    // 绘制虚线
    drawGrid(map_, avp_map.getSemanticElement(SemanticLabel::kDashLine),
             cv::Vec3b(kDashGray, kDashGray, kDashGray));
    // 绘制箭头线
    drawGrid(map_, avp_map.getSemanticElement(SemanticLabel::kArrowLine),
             cv::Vec3b(kArrowGray, kArrowGray, kArrowGray));
    // 启动线程
    thread_ = std::thread([=] { run(); });
  }
  // 显示帧
  void showFrame(const Frame &frame) {
    std::unique_lock<std::mutex> lock(mutex_);
    frame_ = frame;
    ++count_;
  }

 private:
  // 绘制网格（利用珊格化绘制）
  // 珊格的大小怎么确定？
  void drawGrid(const cv::Mat &map,
                const std::unordered_set<Eigen::Vector3i, Vector3iHash> &grid, 
                cv::Vec3b color) const {
    for (const auto &pt : grid) {
      cv::circle(map, offset_ + cv::Point2f(pt.x(), -pt.y()), 1, color, 1);
    }
  }

  // 运行线程
  void run() {
    int cnt = count_;
    cv::Mat show = map_.clone();
    cv::namedWindow("map", cv::WINDOW_NORMAL);
    cv::resizeWindow("map", 300, 200);  // 或按需求设置
    while (cnt > 0) {
      if (cnt != count_) {
        cnt = count_;
        std::unique_lock<std::mutex> lock(mutex_);
        auto frame = std::move(frame_);
        lock.unlock();
        auto pos = to_map_pixel_(frame.t_update);
        if (!trajectory_.empty()) {
          cv::line(map_, trajectory_.back(), pos, cv::Scalar(255, 0, 155), 10);
        }
        trajectory_.push_back(pos);
        show = map_.clone();
        drawGrid(show, frame.getSemanticElement(SemanticLabel::kSlot),
                 cv::Vec3b(0, 0, 255));
        drawGrid(show, frame.getSemanticElement(SemanticLabel::kDashLine),
                 cv::Vec3b(0, 255, 0));
        drawGrid(show, frame.getSemanticElement(SemanticLabel::kArrowLine),
                 cv::Vec3b(255, 0, 0));
      }
      cv::imshow("map", show);
      cv::waitKey(50);
    }
    cv::destroyAllWindows();
  }
  // 线程互斥锁
  std::mutex mutex_;
  // 计数器
  int count_ = 1;
  // 当前帧
  Frame frame_;
  // 线程
  std::thread thread_;

  // 地图图像
  cv::Mat map_;
  // 地图边界框
  Eigen::AlignedBox3d bounding_box_;
  // 偏移量
  cv::Point2f offset_;
  // 轨迹
  std::vector<cv::Point2f> trajectory_;
  // 像素坐标转换函数
  std::function<cv::Point2f(const Eigen::Vector3d &)> to_map_pixel_;
};
