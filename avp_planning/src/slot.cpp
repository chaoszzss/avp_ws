#include "../include/slot.h"

namespace {
double computeCenterDistance(cv::Vec4i line1, cv::Vec4i line2) {
  cv::Point2f p1 ((line1[0] + line1[2]) / 2.0f,
                  (line1[1] + line1[3]) / 2.0f);
  cv::Point2f p2 ((line2[0] + line2[2]) / 2.0f,
                  (line2[1] + line2[3]) / 2.0f);
  return cv::norm(p1 - p2);
}

cv::Point2f computeIntersect(cv::Vec4i line1, cv::Vec4i line2) {
  int x1 = line1[0], y1 = line1[1], x2 = line1[2], y2 = line1[3];
  int x3 = line2[0], y3 = line2[1], x4 = line2[2], y4 = line2[3];
  float denom = (float)((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4));
  cv::Point2f intersect( (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4),
                         (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4));
  return intersect / denom;
}
} // end of namespace

// 保留边角、细小区域 和骨架
cv::Mat skeletonize(const cv::Mat &img) {
  cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
  cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
  cv::Mat temp, eroded;
  do {
    cv::erode(img, eroded, element);
    cv::dilate(eroded, temp, element);
    cv::subtract(img, temp, temp);  // 细小区域
    cv::bitwise_or(skel, temp, skel);
    eroded.copyTo(img);
  } while (0 != cv::countNonZero(img));
  return skel;
}

void removeIsolatedPixels(cv::Mat &src, int min_neighbors) {
  CV_Assert(src.type() == CV_8UC1);  // 确保单通道灰度图
  cv::Mat dst = src.clone();
  for (int y = 1; y < src.rows - 1; ++y) {
    for (int x = 1; x < src.cols - 1; ++x) {
      if (src.at<uchar>(y, x) > 0) {  // 如果是前景图像
        int neighbor_count = 0;       // 相邻前景像素的计数器
        // 检查8邻域内的像素
        for (int ny = -1; ny <= 1; ++ny) {
          for (int nx = -1; nx <= 1; ++nx) {
            if (ny == 0 && nx == 0) continue;  // 跳过自己
            if (src.at<uchar>(y + ny, x + nx) > 0) {
              ++neighbor_count;  // 增加邻居数
            }
          }
        }
        // 如果相邻前景像素小于阈值，则认为该点是孤立的，并将其去除
        if (neighbor_count < min_neighbors) {
          dst.at<uchar>(y, x) = 0;
        }
      }
    }
  }
  src = dst;
}

bool isSlotShortLine(const cv::Point2f &point1,
                     const cv::Point2f &point2,
                     const cv::Mat &image) {
  cv::LineIterator it(image, point1, point2, 4);
  int positiveIndex = 0;
  const double len = cv::norm(point1 - point2);
  int delta = 10;
  if (std::fabs(len - kShortLinePixelDistance) < delta) {
    for (int i = 0; i < it.count; ++i, ++it) {
      int color = image.at<uchar>(std::round(it.pos().y), std::round(it.pos().x));
      if (color > 0) {
        positiveIndex++;
      }
    }
    if (positiveIndex > kShortLineDetectPixelNumberMin) {
      return true;
    }
  }
  return false;
}

bool isSlotLongLine(const cv::Mat &line_img,
                    const cv::Point2f &start,
                    const cv::Point2f &dir) {
  int cnt{0};
  cv::Point2f pt(start);
  for (int l = 1; l < kLongLinePixelDistance; ++l) {
    pt += dir;
    if (pt.y <= 0 || pt.y >= kIPMImgHeight ||
        pt.x <= 0 || pt.x >= kIPMImgWidth) {
      continue;
    }
    if (line_img.at<uchar>(pt) > 0) {
      ++cnt;
    }
  }
  return cnt > kLongLineDetectPixelNumberMin;
}
