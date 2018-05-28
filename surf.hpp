///////////// see LICENSE.txt in the OpenCV root directory //////////////

#ifndef __OPENCV_XFEATURES2D_SURF_HPP__
#define __OPENCV_XFEATURES2D_SURF_HPP__

namespace cv
{
namespace xfeatures2d
{

//! Speeded up robust features, port from CUDA module.
////////////////////////////////// SURF //////////////////////////////////////////
/*!
 SURF implementation.

 The class implements SURF algorithm by H. Bay et al.
 */
class SURF_Impl : public SURF
{
public:
    //! the full constructor taking all the necessary parameters
    explicit CV_WRAP SURF_Impl(double hessianThreshold,
                               int nOctaves = 4, int nOctaveLayers = 2,
                               bool extended = true, bool upright = false);

    //! returns the descriptor size in float's (64 or 128)
    CV_WRAP int descriptorSize() const;

    //! returns the descriptor type
    CV_WRAP int descriptorType() const;

    //! returns the descriptor type
    CV_WRAP int defaultNorm() const;

    void set(int, double);
    double get(int) const;

    //! finds the keypoints and computes their descriptors.
    // Optionally it can compute descriptors for the user-provided keypoints
    void detectAndCompute(InputArray img, InputArray mask,
                          CV_OUT std::vector<KeyPoint>& keypoints,
                          OutputArray descriptors,
                          bool useProvidedKeypoints = false);

    void setHessianThreshold(double hessianThreshold_) { hessianThreshold = hessianThreshold_; }
    double getHessianThreshold() const { return hessianThreshold; }

    void setNOctaves(int nOctaves_) { nOctaves = nOctaves_; }
    int getNOctaves() const { return nOctaves; }

    void setNOctaveLayers(int nOctaveLayers_) { nOctaveLayers = nOctaveLayers_; }
    int getNOctaveLayers() const { return nOctaveLayers; }

    void setExtended(bool extended_) { extended = extended_; }
    bool getExtended() const { return extended; }

    void setUpright(bool upright_) { upright = upright_; }
    bool getUpright() const { return upright; }

    double hessianThreshold;
    int nOctaves;
    int nOctaveLayers;
    bool extended;
    bool upright;
};

/*
template<typename _Tp> void copyVectorToUMat(const std::vector<_Tp>& v, UMat& um)
{
    if(v.empty())
        um.release();
    else
        Mat(1, (int)(v.size()*sizeof(v[0])), CV_8U, (void*)&v[0]).copyTo(um);
}

template<typename _Tp> void copyUMatToVector(const UMat& um, std::vector<_Tp>& v)
{
    if(um.empty())
        v.clear();
    else
    {
        size_t sz = um.total()*um.elemSize();
        CV_Assert(um.isContinuous() && (sz % sizeof(_Tp) == 0));
        v.resize(sz/sizeof(_Tp));
        Mat m(um.size(), um.type(), &v[0]);
        um.copyTo(m);
    }
}*/

}
}

#endif
