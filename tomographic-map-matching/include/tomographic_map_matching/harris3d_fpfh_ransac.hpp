#pragma once

#include <pcl/features/normal_3d.h>
#include <pcl/keypoints/harris_3d.h>
#include <tomographic_map_matching/map_matcher_base.hpp>

namespace map_matcher {

typedef pcl::PointXYZI KeypointT;
typedef pcl::Normal NormalT;
typedef pcl::FPFHSignature33 FeatureT;

typedef pcl::PointCloud<KeypointT> KeypointCloud;
typedef pcl::PointCloud<NormalT> NormalCloud;
typedef pcl::PointCloud<FeatureT> FeatureCloud;

typedef pcl::HarrisKeypoint3D<PointT, KeypointT> Harris3D;

struct Harris3DFPFHRansacParameters : public Parameters {
  Harris3DFPFHRansacParameters() = default;
  Harris3DFPFHRansacParameters(const Harris3DFPFHRansacParameters &) = default;
  Harris3DFPFHRansacParameters(const Parameters &p) : Parameters(p) {}
  float normal_radius = 0.3;
  float fpfh_radius = 0.5;
  float harris_radius = 0.2;
  float harris_corner_threshold = 0.0;
  float ransac_inlier_threshold = 0.1;
  bool ransac_refine_model = true;
};

void to_json(json &j, const Harris3DFPFHRansacParameters &p);
void from_json(const json &j, Harris3DFPFHRansacParameters &p);

class Harris3DFPFHRansac : public MapMatcherBase {
public:
  Harris3DFPFHRansac();
  Harris3DFPFHRansac(Harris3DFPFHRansacParameters parameters);
  json GetParameters() const override;
  void SetParameters(const json &parameters);
  HypothesisPtr RegisterPointCloudMaps(const PointCloud::Ptr pcd1,
                                       const PointCloud::Ptr pcd2,
                                       json &stats) const override;
  void VisualizeKeypoints(const PointCloud::Ptr pcd,
                          const PointCloud::Ptr keypoints) const;

private:
  Harris3DFPFHRansacParameters parameters_;
};

} // namespace map_matcher
