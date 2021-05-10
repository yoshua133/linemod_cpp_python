#include "line2Dup.h"
#include <iostream>

#include<time.h>
#include <string> 

using namespace std;
using namespace cv;

#include <chrono>
class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        std::cout << message << "\nelasped time:" << t << "s\n" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

namespace line2Dup
{
/**
 * \brief Get the label [0,8) of the single bit set in quantized.
 */
static inline int getLabel(int quantized)
{
    switch (quantized)
    {
    case 1:
        return 0;
    case 2:
        return 1;
    case 4:
        return 2;
    case 8:
        return 3;
    case 16:
        return 4;
    case 32:
        return 5;
    case 64:
        return 6;
    case 128:
        return 7;
    default:
        CV_Error(Error::StsBadArg, "Invalid value of quantized parameter");
        return -1; //avoid warning
    }
}

void Feature::read(const FileNode &fn)
{
    FileNodeIterator fni = fn.begin();
    fni >> x >> y >> label;
}

void Feature::write(FileStorage &fs) const
{
    fs << "[:" << x << y << label << "]";
}

void Template::read(const FileNode &fn)
{
    width = fn["width"];
    height = fn["height"];
    tl_x = fn["tl_x"];
    tl_y = fn["tl_y"];
    pyramid_level = fn["pyramid_level"];

    FileNode features_fn = fn["features"];
    features.resize(features_fn.size());
    FileNodeIterator it = features_fn.begin(), it_end = features_fn.end();
    for (int i = 0; it != it_end; ++it, ++i)
    {
        features[i].read(*it);
    }
}

void Template::write(FileStorage &fs) const
{
    fs << "width" << width;
    fs << "height" << height;
    fs << "tl_x" << tl_x;
    fs << "tl_y" << tl_y;
    fs << "pyramid_level" << pyramid_level;

    fs << "features"
       << "[";
    for (int i = 0; i < (int)features.size(); ++i)
    {
        features[i].write(fs);
    }
    fs << "]"; // features
}

static Rect cropTemplates(std::vector<Template> &templates)
{
    int min_x = std::numeric_limits<int>::max();
    int min_y = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int max_y = std::numeric_limits<int>::min();

    // First pass: find min/max feature x,y over all pyramid levels and modalities
    for (int i = 0; i < (int)templates.size(); ++i)
    {
        Template &templ = templates[i];

        for (int j = 0; j < (int)templ.features.size(); ++j)
        {
            int x = templ.features[j].x << templ.pyramid_level;
            int y = templ.features[j].y << templ.pyramid_level;
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
        }
    }

    /// @todo Why require even min_x, min_y?
    if (min_x % 2 == 1)
        --min_x;
    if (min_y % 2 == 1)
        --min_y;

    // Second pass: set width/height and shift all feature positions
    for (int i = 0; i < (int)templates.size(); ++i)
    {
        Template &templ = templates[i];
        templ.width = (max_x - min_x) >> templ.pyramid_level;
        templ.height = (max_y - min_y) >> templ.pyramid_level;
        templ.tl_x = min_x >> templ.pyramid_level;
        templ.tl_y = min_y  >> templ.pyramid_level;

        for (int j = 0; j < (int)templ.features.size(); ++j)
        {
            templ.features[j].x -= templ.tl_x;
            templ.features[j].y -= templ.tl_y;
        }
    }

    return Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}

bool ColorGradientPyramid::selectScatteredFeatures(const std::vector<Candidate> &candidates,
                                                   std::vector<Feature> &features,
                                                   size_t num_features, float distance)
{
    features.clear();
    float distance_sq = distance * distance;
    int i = 0;

    bool first_select = true;

    while(true)
    {
        Candidate c = candidates[i];

        // Add if sufficient distance away from any previously chosen feature
        bool keep = true;
        for (int j = 0; (j < (int)features.size()) && keep; ++j)
        {
            Feature f = features[j];
            keep = (c.f.x - f.x) * (c.f.x - f.x) + (c.f.y - f.y) * (c.f.y - f.y) >= distance_sq;
        }
        if (keep)
            features.push_back(c.f);

        if (++i == (int)candidates.size()){
            bool num_ok = features.size() >= num_features;

            if(first_select){
                if(num_ok){
                    features.clear(); // we don't want too many first time
                    i = 0;
                    distance += 1.0f;
                    distance_sq = distance * distance;
                    continue;
                }else{
                    first_select = false;
                }
            }

            // Start back at beginning, and relax required distance
            i = 0;
            distance -= 1.0f;
            distance_sq = distance * distance;
             if (num_ok || distance < 3){
                 break;
             }
        }
    }
    return true;
}

/****************************************************************************************\
*                                                         Color gradient ColorGradient                                                                        *
\****************************************************************************************/

void hysteresisGradient(Mat &magnitude, Mat &quantized_angle,
                        Mat &angle, float threshold)
{
    // Quantize 360 degree range of orientations into 16 buckets
    // Note that [0, 11.25), [348.75, 360) both get mapped in the end to label 0, ????D:if it is truncated by 0, [0,22,5) -> 0 [180,202.5)->8]
    // for stability of horizontal and vertical features.
    Mat_<unsigned char> quantized_unfiltered;
    angle.convertTo(quantized_unfiltered, CV_8U, 16.0 / 360.0);

    // Zero out top and bottom rows
    /// @todo is this necessary, or even correct?
    memset(quantized_unfiltered.ptr(), 0, quantized_unfiltered.cols);
    memset(quantized_unfiltered.ptr(quantized_unfiltered.rows - 1), 0, quantized_unfiltered.cols);
    // Zero out first and last columns
    for (int r = 0; r < quantized_unfiltered.rows; ++r)
    {
        quantized_unfiltered(r, 0) = 0;
        quantized_unfiltered(r, quantized_unfiltered.cols - 1) = 0;
    }

    // Mask 16 buckets into 8 quantized orientations
    for (int r = 1; r < angle.rows - 1; ++r)
    {
        uchar *quant_r = quantized_unfiltered.ptr<uchar>(r);
        for (int c = 1; c < angle.cols - 1; ++c)
        {
            quant_r[c] &= 7;
        }
    }

    // Filter the raw quantized image. Only accept pixels where the magnitude is above some
    // threshold, and there is local agreement on the quantization.
    quantized_angle = Mat::zeros(angle.size(), CV_8U);
    for (int r = 1; r < angle.rows - 1; ++r)
    {
        float *mag_r = magnitude.ptr<float>(r);

        for (int c = 1; c < angle.cols - 1; ++c)
        {
            if (mag_r[c] > threshold)
            {
                // Compute histogram of quantized bins in 3x3 patch around pixel
                int histogram[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                uchar *patch3x3_row = &quantized_unfiltered(r - 1, c - 1);
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                patch3x3_row += quantized_unfiltered.step1();
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                patch3x3_row += quantized_unfiltered.step1();
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                // Find bin with the most votes from the patch
                int max_votes = 0;
                int index = -1;
                for (int i = 0; i < 8; ++i)
                {
                    if (max_votes < histogram[i])
                    {
                        index = i;
                        max_votes = histogram[i];
                    }
                }

                // Only accept the quantization if majority of pixels in the patch agree
                static const int NEIGHBOR_THRESHOLD = 5;
                if (max_votes >= NEIGHBOR_THRESHOLD)
                    quantized_angle.at<uchar>(r, c) = uchar(1 << index);
            }
        }
    }
}

static void quantizedOrientations(const Mat &src, Mat &magnitude,
                                  Mat &angle, Mat& angle_ori, float threshold)
{
    Mat smoothed;
    // Compute horizontal and vertical image derivatives on all color channels separately
    static const int KERNEL_SIZE = 7;
    // For some reason cvSmooth/cv::GaussianBlur, cvSobel/cv::Sobel have different defaults for border handling...
    GaussianBlur(src, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);

    if(src.channels() == 1){
        Mat sobel_dx, sobel_dy, sobel_ag;
        Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
        Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
        magnitude = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
        phase(sobel_dx, sobel_dy, sobel_ag, true);
        hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);
        angle_ori = sobel_ag;

    }else{

        magnitude.create(src.size(), CV_32F);

        // Allocate temporary buffers
        Size size = src.size();
        Mat sobel_3dx;              // per-channel horizontal derivative
        Mat sobel_3dy;              // per-channel vertical derivative
        Mat sobel_dx(size, CV_32F); // maximum horizontal derivative
        Mat sobel_dy(size, CV_32F); // maximum vertical derivative
        Mat sobel_ag;               // final gradient orientation (unquantized)

        Sobel(smoothed, sobel_3dx, CV_16S, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
        Sobel(smoothed, sobel_3dy, CV_16S, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);

        short *ptrx = (short *)sobel_3dx.data;
        short *ptry = (short *)sobel_3dy.data;
        float *ptr0x = (float *)sobel_dx.data;
        float *ptr0y = (float *)sobel_dy.data;
        float *ptrmg = (float *)magnitude.data;

        const int length1 = static_cast<const int>(sobel_3dx.step1());
        const int length2 = static_cast<const int>(sobel_3dy.step1());
        const int length3 = static_cast<const int>(sobel_dx.step1());
        const int length4 = static_cast<const int>(sobel_dy.step1());
        const int length5 = static_cast<const int>(magnitude.step1());
        const int length0 = sobel_3dy.cols * 3;

        for (int r = 0; r < sobel_3dy.rows; ++r)
        {
            int ind = 0;

            for (int i = 0; i < length0; i += 3)
            {
                // Use the gradient orientation of the channel whose magnitude is largest
                int mag1 = ptrx[i + 0] * ptrx[i + 0] + ptry[i + 0] * ptry[i + 0];
                int mag2 = ptrx[i + 1] * ptrx[i + 1] + ptry[i + 1] * ptry[i + 1];
                int mag3 = ptrx[i + 2] * ptrx[i + 2] + ptry[i + 2] * ptry[i + 2];

                if (mag1 >= mag2 && mag1 >= mag3)
                {
                    ptr0x[ind] = ptrx[i];
                    ptr0y[ind] = ptry[i];
                    ptrmg[ind] = (float)mag1;
                }
                else if (mag2 >= mag1 && mag2 >= mag3)
                {
                    ptr0x[ind] = ptrx[i + 1];
                    ptr0y[ind] = ptry[i + 1];
                    ptrmg[ind] = (float)mag2;
                }
                else
                {
                    ptr0x[ind] = ptrx[i + 2];
                    ptr0y[ind] = ptry[i + 2];
                    ptrmg[ind] = (float)mag3;
                }
                ++ind;
            }
            ptrx += length1;
            ptry += length2;
            ptr0x += length3;
            ptr0y += length4;
            ptrmg += length5;
        }

        // Calculate the final gradient orientations
        phase(sobel_dx, sobel_dy, sobel_ag, true);
        hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);
        angle_ori = sobel_ag;
    }


}

ColorGradientPyramid::ColorGradientPyramid(const Mat &_src, const Mat &_mask,
                                           float _weak_threshold, size_t _num_features,
                                           float _strong_threshold)
    : src(_src),
      mask(_mask),
      pyramid_level(0),
      weak_threshold(_weak_threshold),
      num_features(_num_features),
      strong_threshold(_strong_threshold)
{
    update();
}

void ColorGradientPyramid::update()
{
    quantizedOrientations(src, magnitude, angle, angle_ori, weak_threshold);
}

void ColorGradientPyramid::pyrDown()
{
    // Some parameters need to be adjusted
    num_features /= 2; /// @todo Why not 4?
    ++pyramid_level;

    // Downsample the current inputs
    Size size(src.cols / 2, src.rows / 2);
    Mat next_src;
    cv::pyrDown(src, next_src, size);
    src = next_src;

    if (!mask.empty())
    {
        Mat next_mask;
        resize(mask, next_mask, size, 0.0, 0.0, INTER_NEAREST);
        mask = next_mask;
    }

    update();
}

void ColorGradientPyramid::quantize(Mat &dst) const
{
    dst = Mat::zeros(angle.size(), CV_8U);
    angle.copyTo(dst, mask);
}

bool ColorGradientPyramid::extractTemplate(Template &templ) const
{
    // Want features on the border to distinguish from background
    Mat local_mask;
    if (!mask.empty())
    {
        erode(mask, local_mask, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
//        subtract(mask, local_mask, local_mask);
    }

    std::vector<Candidate> candidates;
    bool no_mask = local_mask.empty();
    float threshold_sq = strong_threshold * strong_threshold;

    int nms_kernel_size = 5;
    cv::Mat magnitude_valid = cv::Mat(magnitude.size(), CV_8UC1, cv::Scalar(255));

    for (int r = 0+nms_kernel_size/2; r < magnitude.rows-nms_kernel_size/2; ++r)
    {
        const uchar *mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);

        for (int c = 0+nms_kernel_size/2; c < magnitude.cols-nms_kernel_size/2; ++c)
        {
            if (no_mask || mask_r[c])
            {
                float score = 0;
                if(magnitude_valid.at<uchar>(r, c)>0){
                    score = magnitude.at<float>(r, c);
                    bool is_max = true;
                    for(int r_offset = -nms_kernel_size/2; r_offset <= nms_kernel_size/2; r_offset++){
                        for(int c_offset = -nms_kernel_size/2; c_offset <= nms_kernel_size/2; c_offset++){
                            if(r_offset == 0 && c_offset == 0) continue;

                            if(score < magnitude.at<float>(r+r_offset, c+c_offset)){
                                score = 0;
                                is_max = false;
                                break;
                            }
                        }
                        if(!is_max) break;
                    }

                    if(is_max){
                        for(int r_offset = -nms_kernel_size/2; r_offset <= nms_kernel_size/2; r_offset++){
                            for(int c_offset = -nms_kernel_size/2; c_offset <= nms_kernel_size/2; c_offset++){
                                if(r_offset == 0 && c_offset == 0) continue;
                                magnitude_valid.at<uchar>(r+r_offset, c+c_offset) = 0;
                            }
                        }
                    }
                }

                if (score > threshold_sq && angle.at<uchar>(r, c) > 0)
                {
                    candidates.push_back(Candidate(c, r, getLabel(angle.at<uchar>(r, c)), score));
                    candidates.back().f.theta = angle_ori.at<float>(r, c);
                }
            }
        }
    }
    // We require a certain number of features
    if (candidates.size() < num_features){
        if(candidates.size() <= 4) {
            std::cout << "too few features, abort" << std::endl;
            return false;
        }
        std::cout << "have no enough features, exaustive mode" << std::endl;
    }

    // NOTE: Stable sort to agree with old code, which used std::list::sort()
    std::stable_sort(candidates.begin(), candidates.end());

    // Use heuristic based on surplus of candidates in narrow outline for initial distance threshold
    float distance = static_cast<float>(candidates.size() / num_features + 1);

    // selectScatteredFeatures always return true
    if (!selectScatteredFeatures(candidates, templ.features, num_features, distance))  // this is absolutely return true
    {
        return false;
    }

    // Size determined externally, needs to match templates for other modalities
    templ.width = -1;
    templ.height = -1;
    templ.pyramid_level = pyramid_level;

    return true;
}

ColorGradient::ColorGradient()
    : weak_threshold(30.0f),
      num_features(63),
      strong_threshold(60.0f)
{
}

ColorGradient::ColorGradient(float _weak_threshold, size_t _num_features, float _strong_threshold)
    : weak_threshold(_weak_threshold),
      num_features(_num_features),
      strong_threshold(_strong_threshold)
{
}

static const char CG_NAME[] = "ColorGradient";

std::string ColorGradient::name() const
{
    return CG_NAME;
}

void ColorGradient::read(const FileNode &fn)
{
    String type = fn["type"];
    CV_Assert(type == CG_NAME);

    weak_threshold = fn["weak_threshold"];
    num_features = int(fn["num_features"]);
    strong_threshold = fn["strong_threshold"];
}

void ColorGradient::write(FileStorage &fs) const
{
    fs << "type" << CG_NAME;
    fs << "weak_threshold" << weak_threshold;
    fs << "num_features" << int(num_features);
    fs << "strong_threshold" << strong_threshold;
}
/****************************************************************************************\
*                                                                 Response maps                                                                                    *
\****************************************************************************************/

static void orUnaligned8u(const uchar *src, const int src_stride,
                          uchar *dst, const int dst_stride,
                          const int width, const int height)
{
    for (int r = 0; r < height; ++r)
    {
        int c = 0;
        // why not just merge into one loop??

        // not aligned, which will happen because we move 1 bytes a time for spreading
        while (reinterpret_cast<unsigned long long>(src + c) % 16 != 0) {
            dst[c] |= src[c];  // Or operation and then equal
            c++;
        }

        // avoid out of bound when can't divid
        // note: can't use c<width !!!
        for (; c <= width-mipp::N<uint8_t>(); c+=mipp::N<uint8_t>()){
            mipp::Reg<uint8_t> src_v((uint8_t*)src + c);
            mipp::Reg<uint8_t> dst_v((uint8_t*)dst + c); //D:guess this means src_v =  src[c] and res_v = src_v | dst_v
            // then dst[c] = res_v

            mipp::Reg<uint8_t> res_v = mipp::orb(src_v, dst_v); // orb is also OR operation
            res_v.store((uint8_t*)dst + c);
        }

        for(; c<width; c++)
            dst[c] |= src[c];

        // Advance to next row
        src += src_stride;
        dst += dst_stride;
    }
}

static void spread(const Mat &src, Mat &dst, int T)
{
    // Allocate and zero-initialize spread (OR'ed) image
    dst = Mat::zeros(src.size(), CV_8U);

    // Fill in spread gradient image (section 2.3) T=4,8
    for (int r = 0; r < T; ++r)
    {
        for (int c = 0; c < T; ++c)
        {
            orUnaligned8u(&src.at<unsigned char>(r, c), static_cast<const int>(src.step1()), dst.ptr(),
                          static_cast<const int>(dst.step1()), src.cols - c, src.rows - r);
        }
    }
}

static const unsigned char LUT3 = 2;
// 1,2-->0 3-->LUT3
CV_DECL_ALIGNED(16)
static const unsigned char SIMILARITY_LUT[256] = {0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, 4, 4, 4, 4, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, 4, 4, 4, 4, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4};
inline int com_simi(int ori, uchar ori_bits){
    if (((ori_bits >> ori) & 1) ==1){
        return 4;
    }
    if (ori ==0){
        if ( (((ori_bits >> 1) & 1) ==1) || (((ori_bits >> 7) & 1) ==1) ){
            return LUT3;
        }
        else{return 0;}
    }
    else if (ori ==7){
        if ( (((ori_bits >> 6) & 1) ==1) || ((ori_bits  & 1) ==1) ){
            return LUT3;
        }
        else{return 0;}

    }
    else{
        if ( (((ori_bits >> (ori-1))& 1) ==1) || (((ori_bits >> (ori+1)) & 1) ==1) ){
            return LUT3;
        }
        else{return 0;}

    }
}


static void computeResponseMaps(const Mat &src, std::vector<Mat> &response_maps)
{
    Timer timer3;
    CV_Assert((src.rows * src.cols) % 16 == 0);

    // Allocate response maps
    response_maps.resize(8);
    for (int i = 0; i < 8; ++i)
        response_maps[i].create(src.size(), CV_8U);
    
    const uchar *src_data = src.ptr<uchar>();

    //Mat lsb4(src.size(), CV_8U);
    // Mat msb4(src.size(), CV_8U);

    // for (int r = 0; r < src.rows; ++r)
    // {
    //     const uchar *src_r = src.ptr(r);
    //     uchar *lsb4_r = lsb4.ptr(r);
    //     uchar *msb4_r = msb4.ptr(r);

    //     for (int c = 0; c < src.cols; ++c) //D: lsb:coding whether the latter 4 orientation in the pixel, msb:the ahead 4.
    //     {
    //         // Least significant 4 bits of spread image pixel
    //         lsb4_r[c] = src_r[c] & 15; // and operation, means find a small number of 15 and 
    //         // Most significant 4 bits, right-shifted to be in [0, 16)
    //         msb4_r[c] = (src_r[c] & 240) >> 4; //this means divided by 2^4=16
    //     }
    // }

    {
        // uchar *lsb4_data = lsb4.ptr<uchar>();
        // uchar *msb4_data = msb4.ptr<uchar>();

        bool no_max = true;
        bool no_shuff = true;

#ifdef has_max_int8_t
        no_max = false;
#endif

#ifdef has_shuff_int8_t
        no_shuff = false;
#endif
        // LUT is designed for 128 bits SIMD, so quite triky for others
        //timer3.out("create response map");
        // For each of the 8 quantized orientations...
        for (int ori = 0; ori < 8; ++ori){
            uchar *map_data = response_maps[ori].ptr<uchar>();
            // const uchar *lut_low = SIMILARITY_LUT + 32 * ori;

            
            for (int i = 0; i < src.rows * src.cols; ++i)
                map_data[i] = com_simi(ori, src_data[i]);//std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);
            std::string text = "one time ";
            text += std::to_string(ori);
            //timer3.out(text);
            // else if(mipp::N<uint8_t>() == 16){ // 128 SIMD, no add base

            //     // const uchar *lut_low = SIMILARITY_LUT + 32 * ori;
            //     // mipp::Reg<uint8_t> lut_low_v((uint8_t*)lut_low);
            //     // mipp::Reg<uint8_t> lut_high_v((uint8_t*)lut_low + 16);

            //     for (int i = 0; i < src.rows * src.cols; i += mipp::N<uint8_t>()){
            //         mipp::Reg<uint8_t> low_mask((uint8_t*)lsb4_data + i);
            //         mipp::Reg<uint8_t> high_mask((uint8_t*)msb4_data + i);

            //         mipp::Reg<uint8_t> low_res = mipp::shuff(lut_low_v, low_mask);
            //         mipp::Reg<uint8_t> high_res = mipp::shuff(lut_high_v, high_mask);

            //         mipp::Reg<uint8_t> result = mipp::max(low_res, high_res);
            //         result.store((uint8_t*)map_data + i);
            //     }
            // }
            // else if(mipp::N<uint8_t>() == 16 || mipp::N<uint8_t>() == 32
            //         || mipp::N<uint8_t>() == 64){ //128 256 512 SIMD
            //     CV_Assert((src.20210430_000935part0_apr18_revised_crop1_725_aug_p_0_attri_9resnet_101pretrain-Falsesize224(lut_low, 16, lut_temp+slice*16);
            //     }
            //     mipp::Reg<uint8_t> lut_low_v(lut_temp);

            //     uint8_t base_add_array[mipp::N<uint8_t>()] = {0};
            //     for(uint8_t slice=0; slice<mipp::N<uint8_t>(); slice+=16){
            //         std::copy_n(lut_low+16, 16, lut_temp+slice);
            //         std::fill_n(base_add_array+slice, 16, slice);
            //     }Timerg<uint8_t> mask_low_v((uint8_t*)lsb4_data+i);
            //         mipp::Reg<uint8_t> mask_high_v((uint8_t*)msb4_data+i);

            //         mask_low_v += base_add;
            //         mask_high_v += base_add;

            //         mipp::Reg<uint8_t> shuff_low_result = mipp::shuff(lut_low_v, mask_low_v);
            //         mipp::Reg<uint8_t> shuff_high_result = mipp::shuff(lut_high_v, mask_high_v);

            //         mipp::Reg<uint8_t> result = mipp::max(shuff_low_result, shuff_high_result);
            //         result.store((uint8_t*)map_data + i);
            //         }
            //     }
            // else{
            //     for (int i = 0; i < src.rows * src.cols; ++i)
            //         //map_data[i] = std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);
            //         map_data[i] = com_simi(ori, src_data[i])            
            // }
        }


    }
}





// below is the original compute

// static void computeResponseMaps(const Mat &src, std::vector<Mat> &response_maps)
// {
//     CV_Assert((src.rows * src.cols) % 16 == 0);

//     // Allocate response maps
//     response_maps.resize(8);
//     for (int i = 0; i < 8; ++i)
//         response_maps[i].create(src.size(), CV_8U);

//     Mat lsb4(src.size(), CV_8U);
//     Mat msb4(src.size(), CV_8U);

//     for (int r = 0; r < src.rows; ++r)
//     {
//         const uchar *src_r = src.ptr(r);
//         uchar *lsb4_r = lsb4.ptr(r);
//         uchar *msb4_r = msb4.ptr(r);

//         for (int c = 0; c < src.cols; ++c) //D: lsb:coding whether the latter 4 orientation in the pixel, msb:the ahead 4.
//         {
//             // Least significant 4 bits of spread image pixel
//             lsb4_r[c] = src_r[c] & 15; // and operation, means find a small number of 15 and 
//             // Most significant 4 bits, right-shifted to be in [0, 16)
//             msb4_r[c] = (src_r[c] & 240) >> 4; //this means divided by 2^4=16
//         }
//     }

//     {
//         uchar *lsb4_data = lsb4.ptr<uchar>();
//         uchar *msb4_data = msb4.ptr<uchar>();

//         bool no_max = true;
//         bool no_shuff = true;

// #ifdef has_max_int8_t
//         no_max = false;
// #endif

// #ifdef has_shuff_int8_t
//         no_shuff = false;
// #endif
//         // LUT is designed for 128 bits SIMD, so quite triky for others

//         // For each of the 8 quantized orientations...
//         for (int ori = 0; ori < 8; ++ori){
//             uchar *map_data = response_maps[ori].ptr<uchar>();
//             const uchar *lut_low = SIMILARITY_LUT + 32 * ori;

//             if(mipp::N<uint8_t>() == 1 || no_max || no_shuff){ // no SIMD
//                 for (int i = 0; i < src.rows * src.cols; ++i)
//                     map_data[i] = std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);
//             }
//             else if(mipp::N<uint8_t>() == 16){ // 128 SIMD, no add base

//                 const uchar *lut_low = SIMILARITY_LUT + 32 * ori;
//                 mipp::Reg<uint8_t> lut_low_v((uint8_t*)lut_low);
//                 mipp::Reg<uint8_t> lut_high_v((uint8_t*)lut_low + 16);

//                 for (int i = 0; i < src.rows * src.cols; i += mipp::N<uint8_t>()){
//                     mipp::Reg<uint8_t> low_mask((uint8_t*)lsb4_data + i);
//                     mipp::Reg<uint8_t> high_mask((uint8_t*)msb4_data + i);

//                     mipp::Reg<uint8_t> low_res = mipp::shuff(lut_low_v, low_mask);
//                     mipp::Reg<uint8_t> high_res = mipp::shuff(lut_high_v, high_mask);

//                     mipp::Reg<uint8_t> result = mipp::max(low_res, high_res);
//                     result.store((uint8_t*)map_data + i);
//                 }
//             }
//             else if(mipp::N<uint8_t>() == 16 || mipp::N<uint8_t>() == 32
//                     || mipp::N<uint8_t>() == 64){ //128 256 512 SIMD
//                 CV_Assert((src.rows * src.cols) % mipp::N<uint8_t>() == 0);

//                 uint8_t lut_temp[mipp::N<uint8_t>()] = {0};

//                 for(int slice=0; slice<mipp::N<uint8_t>()/16; slice++){
//                     std::copy_n(lut_low, 16, lut_temp+slice*16);
//                 }
//                 mipp::Reg<uint8_t> lut_low_v(lut_temp);

//                 uint8_t base_add_array[mipp::N<uint8_t>()] = {0};
//                 for(uint8_t slice=0; slice<mipp::N<uint8_t>(); slice+=16){
//                     std::copy_n(lut_low+16, 16, lut_temp+slice);
//                     std::fill_n(base_add_array+slice, 16, slice);
//                 }
//                 mipp::Reg<uint8_t> base_add(base_add_array);
//                 mipp::Reg<uint8_t> lut_high_v(lut_temp);

//                 for (int i = 0; i < src.rows * src.cols; i += mipp::N<uint8_t>()){
//                     mipp::Reg<uint8_t> mask_low_v((uint8_t*)lsb4_data+i);
//                     mipp::Reg<uint8_t> mask_high_v((uint8_t*)msb4_data+i);

//                     mask_low_v += base_add;
//                     mask_high_v += base_add;

//                     mipp::Reg<uint8_t> shuff_low_result = mipp::shuff(lut_low_v, mask_low_v);
//                     mipp::Reg<uint8_t> shuff_high_result = mipp::shuff(lut_high_v, mask_high_v);

//                     mipp::Reg<uint8_t> result = mipp::max(shuff_low_result, shuff_high_result);
//                     result.store((uint8_t*)map_data + i);
//                     }
//                 }
//             else{
//                 for (int i = 0; i < src.rows * src.cols; ++i)
//                     map_data[i] = std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);
//             }
//         }


//     }
// }

static void linearize(const Mat &response_map, Mat &linearized, int T)
{
    CV_Assert(response_map.rows % T == 0);
    CV_Assert(response_map.cols % T == 0);

    // linearized has T^2 rows, where each row is a linear memory
    int mem_width = response_map.cols / T;
    int mem_height = response_map.rows / T;
    linearized.create(T * T, mem_width * mem_height, CV_8U);

    // Outer two for loops iterate over top-left T^2 starting pixels
    int index = 0;
    for (int r_start = 0; r_start < T; ++r_start)
    {
        for (int c_start = 0; c_start < T; ++c_start)
        {
            uchar *memory = linearized.ptr(index);
            ++index;

            // Inner two loops copy every T-th pixel into the linear memory
            for (int r = r_start; r < response_map.rows; r += T)
            {
                const uchar *response_data = response_map.ptr(r);
                for (int c = c_start; c < response_map.cols; c += T)
                    *memory++ = response_data[c];// memory++ and then memory = response_data
            }
        }
    }
}
/****************************************************************************************\
*                                                             Linearized similarities                                                                    *
\****************************************************************************************/

static const unsigned char *accessLinearMemory(const std::vector<Mat> &linear_memories,
                                               const Feature &f, int T, int W)
{
    // Retrieve the TxT grid of linear memories associated with the feature label
    const Mat &memory_grid = linear_memories[f.label];
    CV_DbgAssert(memory_grid.rows == T * T);
    CV_DbgAssert(f.x >= 0);
    CV_DbgAssert(f.y >= 0);
    // The LM we want is at (x%T, y%T) in the TxT grid (stored as the rows of memory_grid)
    int grid_x = f.x % T;
    int grid_y = f.y % T;
    int grid_index = grid_y * T + grid_x;
    CV_DbgAssert(grid_index >= 0);
    CV_DbgAssert(grid_index < memory_grid.rows);
    const unsigned char *memory = memory_grid.ptr(grid_index);
    // Within the LM, the feature is at (x/T, y/T). W is the "width" of the LM, the
    // input image width decimated by T.
    int lm_x = f.x / T;
    int lm_y = f.y / T;
    int lm_index = lm_y * W + lm_x;
    CV_DbgAssert(lm_index >= 0);
    CV_DbgAssert(lm_index < memory_grid.cols);
    return memory + lm_index;
}

static void similarity(const std::vector<Mat> &linear_memories, const Template &templ,
                       Mat &dst, Size size, int T)
{
    // we only have one modality, so 8192*2, due to mipp, back to 8192
    CV_Assert(templ.features.size() < 8192);

    // Decimate input image size by factor of T
    int W = size.width / T;
    int H = size.height / T;

    // Feature dimensions, decimated by factor T and rounded up
    int wf = (templ.width - 1) / T + 1;
    int hf = (templ.height - 1) / T + 1;

    // Span is the range over which we can shift the template around the input image
    int span_x = W - wf;
    int span_y = H - hf;

    int template_positions = span_y * W + span_x + 1; // why add 1?

    dst = Mat::zeros(H, W, CV_16U);
    short *dst_ptr = dst.ptr<short>();
    mipp::Reg<uint8_t> zero_v(uint8_t(0));

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {

        Feature f = templ.features[i];

        if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
            continue;
        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        int j = 0;

        // *2 to avoid int8 read out of range
        for(; j <= template_positions -mipp::N<int16_t>()*2; j+=mipp::N<int16_t>()){
            mipp::Reg<uint8_t> src8_v((uint8_t*)lm_ptr + j);

            // uchar to short, once for N bytes
            mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

            mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + j);

            mipp::Reg<int16_t> res_v = src16_v + dst_v;
            res_v.store((int16_t*)dst_ptr + j);
        }

        for(; j<template_positions; j++)
            dst_ptr[j] += short(lm_ptr[j]);
    }
}

static void similarityLocal(const std::vector<Mat> &linear_memories, const Template &templ,
                            Mat &dst, Size size, int T, Point center)
{
    CV_Assert(templ.features.size() < 8192);

    int W = size.width / T;
    dst = Mat::zeros(16, 16, CV_16U);

    int offset_x = (center.x / T - 8) * T;
    int offset_y = (center.y / T - 8) * T;
    mipp::Reg<uint8_t> zero_v = uint8_t(0);

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        Feature f = templ.features[i];
        f.x += offset_x;
        f.y += offset_y;
        // Discard feature if out of bounds, possibly due to applying the offset
        if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
            continue;

        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);
        {
            short *dst_ptr = dst.ptr<short>();

            if(mipp::N<uint8_t>() > 32){ //512 bits SIMD
                for (int row = 0; row < 16; row += mipp::N<int16_t>()/16){
                    mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + row*16);

                    // load lm_ptr, 16 bytes once, for half
                    uint8_t local_v[mipp::N<uint8_t>()] = {0};
                    for(int slice=0; slice<mipp::N<uint8_t>()/16/2; slice++){
                        std::copy_n(lm_ptr, 16, &local_v[16*slice]);
                        lm_ptr += W;
                    }
                    mipp::Reg<uint8_t> src8_v(local_v);
                    // uchar to short, once for N bytes
                    mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

                    mipp::Reg<int16_t> res_v = src16_v + dst_v;
                    res_v.store((int16_t*)dst_ptr);

                    dst_ptr += mipp::N<int16_t>();
                }
            }else{ // 256 128 or no SIMD
                for (int row = 0; row < 16; ++row){
                    for(int col=0; col<16; col+=mipp::N<int16_t>()){
                        mipp::Reg<uint8_t> src8_v((uint8_t*)lm_ptr + col);

                        // uchar to short, once for N bytes
                        mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

                        mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + col);
                        mipp::Reg<int16_t> res_v = src16_v + dst_v;
                        res_v.store((int16_t*)dst_ptr + col);
                    }
                    dst_ptr += 16;
                    lm_ptr += W;
                }
            }
        }
    }
}

static void similarity_64(const std::vector<Mat> &linear_memories, const Template &templ,
                          Mat &dst, Size size, int T)
{
    // 63 features or less is a special case because the max similarity per-feature is 4.
    // 255/4 = 63, so up to that many we can add up similarities in 8 bits without worrying
    // about overflow. Therefore here we use _mm_add_epi8 as the workhorse, whereas a more
    // general function would use _mm_add_epi16.
    CV_Assert(templ.features.size() < 64);
    /// @todo Handle more than 255/MAX_RESPONSE features!!

    // Decimate input image size by factor of T
    int W = size.width / T;   //size is the input image size
    int H = size.height / T;

    // Feature dimensions, decimated by factor T and rounded up
    int wf = (templ.width - 1) / T + 1;
    int hf = (templ.height - 1) / T + 1;

    // Span is the range over which we can shift the template around the input image
    int span_x = W - wf;
    int span_y = H - hf;

    // Compute number of contiguous (in memory) pixels to check when sliding feature over
    // image. This allows template to wrap around left/right border incorrectly, so any
    // wrapped template matches must be filtered out!
    int template_positions = span_y * W + span_x + 1; // why add 1?
    //int template_positions = (span_y - 1) * W + span_x; // More correct?
    // D:???? why not (span_x+1)*(span_y+1)

    /// @todo In old code, dst is buffer of size m_U. Could make it something like
    /// (span_x)x(span_y) instead?
    dst = Mat::zeros(H, W, CV_8U);
    uchar *dst_ptr = dst.ptr<uchar>();

    // Compute the similarity measure for this template by accumulating the contribution of
    // each feature
    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        // Add the linear memory at the appropriate offset computed from the location of
        // the feature in the template
        Feature f = templ.features[i];
        // Discard feature if out of bounds
        /// @todo Shouldn't actually see x or y < 0 here?
        if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
            continue;
        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        // Now we do an aligned/unaligned add of dst_ptr and lm_ptr with template_positions elements
        int j = 0;

        for(; j <= template_positions -mipp::N<uint8_t>(); j+=mipp::N<uint8_t>()){
            mipp::Reg<uint8_t> src_v((uint8_t*)lm_ptr + j);
            mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr + j);

            mipp::Reg<uint8_t> res_v = src_v + dst_v;
            res_v.store((uint8_t*)dst_ptr + j);
        }

        for(; j<template_positions; j++)
            dst_ptr[j] += lm_ptr[j];
    }
}

static void similarityLocal_64(const std::vector<Mat> &linear_memories, const Template &templ,
                               Mat &dst, Size size, int T, Point center)
{
    // Similar to whole-image similarity() above. This version takes a position 'center'
    // and computes the energy in the 16x16 patch centered on it.
    CV_Assert(templ.features.size() < 64);

    // Compute the similarity map in a 16x16 patch around center
    int W = size.width / T;
    dst = Mat::zeros(16, 16, CV_8U);

    // Offset each feature point by the requested center. Further adjust to (-8,-8) from the
    // center to get the top-left corner of the 16x16 patch.
    // NOTE: We make the offsets multiples of T to agree with results of the original code.
    int offset_x = (center.x / T - 8) * T;
    int offset_y = (center.y / T - 8) * T;

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        Feature f = templ.features[i];
        f.x += offset_x;
        f.y += offset_y;
        // Discard feature if out of bounds, possibly due to applying the offset
        if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
            continue;

        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        {
            uchar *dst_ptr = dst.ptr<uchar>();

            if(mipp::N<uint8_t>() > 16){ // 256 or 512 bits SIMD
                for (int row = 0; row < 16; row += mipp::N<uint8_t>()/16){
                    mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr);

                    // load lm_ptr, 16 bytes once
                    uint8_t local_v[mipp::N<uint8_t>()];
                    for(int slice=0; slice<mipp::N<uint8_t>()/16; slice++){
                        std::copy_n(lm_ptr, 16, &local_v[16*slice]);
                        lm_ptr += W;
                    }
                    mipp::Reg<uint8_t> src_v(local_v);

                    mipp::Reg<uint8_t> res_v = src_v + dst_v;
                    res_v.store((uint8_t*)dst_ptr);

                    dst_ptr += mipp::N<uint8_t>();
                }
            }else{ // 128 or no SIMD
                for (int row = 0; row < 16; ++row){
                    for(int col=0; col<16; col+=mipp::N<uint8_t>()){
                        mipp::Reg<uint8_t> src_v((uint8_t*)lm_ptr + col);
                        mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr + col);
                        mipp::Reg<uint8_t> res_v = src_v + dst_v;
                        res_v.store((uint8_t*)dst_ptr + col);
                    }
                    dst_ptr += 16;
                    lm_ptr += W;
                }
            }
        }
    }
}

/****************************************************************************************\
*                                                             High-level Detector API                                                                    *
\****************************************************************************************/

Detector::Detector()
{
    this->modality = makePtr<ColorGradient>();
    pyramid_levels = 2;
    T_at_level.push_back(4);
    T_at_level.push_back(8);
}

Detector::Detector(std::vector<int> T)
{
    this->modality = makePtr<ColorGradient>();
    pyramid_levels = T.size();
    T_at_level = T;
}

string Replace(string& str, const string& sub, const string& mod) {
    string tmp(str);
    tmp.replace(tmp.find(sub), mod.length(), mod);
    return tmp;
}

void save(string name_i, std::vector<Mat> &data){
    for (int j =0;j<data.size();j++){
        string name_j = Replace(name_i, "spread", "_"+std::to_string(j)+"_response.xml");
        string name_j_img = Replace(name_i, "spread", "_"+std::to_string(j)+"_response.png");
        //cout<< "name_j" <<name_j;
        //cout<< "name_j_img" << name_j_img;
        FileStorage fs(name_j, FileStorage::WRITE);
        fs<<"vocabulary"<< data[j];
        fs.release();
        cv::imwrite(name_j_img, data[j]);
    }
}


void save_single(string name_i, Mat &data){
    FileStorage fs(name_i, FileStorage::WRITE);
    fs<<"vocabulary"<< data;
    fs.release();
    cv::imwrite(Replace(name_i, "xml","png"), data);
    
}




Detector::Detector(int num_features, std::vector<int> T, float weak_thresh, float strong_threash)
{
    this->modality = makePtr<ColorGradient>(weak_thresh, num_features, strong_threash);
    pyramid_levels = T.size();
    T_at_level = T;
}
string prefix = "/home/xiangdawei/linemod_python/linemod_cpp_python/result_visual/";
std::vector<Match> Detector::match(Mat source, float threshold, std::string name_i ,
                                   const std::vector<std::string> &class_ids, const Mat mask) const
{
    Timer timer;
    Timer timer2;
    std::vector<Match> matches;
    cout << "match name-i   "<< name_i;
    cout << "mat shape" << source.size();
    // Initialize each ColorGradient with our sources
    std::vector<Ptr<ColorGradientPyramid>> quantizers;
    CV_Assert(mask.empty() || mask.size() == source.size());
    quantizers.push_back(modality->process(source, mask));
    //timer2.out("process");
    // pyramid level -> ColorGradient -> quantization
    LinearMemoryPyramid lm_pyramid(pyramid_levels,
                                   std::vector<LinearMemories>(1, LinearMemories(8)));

    // For each pyramid level, precompute linear memories for each ColorGradient
    std::vector<Size> sizes;
    //cout<<"pyramid_levels"<<pyramid_levels; // 2
    //timer2.out("lm pyramids");
    //Mat quantized, spread_quantized;
    std::vector<Mat> response_maps;
    for (int l = 0; l < pyramid_levels; ++l)
    {
        int T =  T_at_level[l];
        //cout<< "T_at_level[l]" << T_at_level[l]; //4,8
        std::vector<LinearMemories> &lm_level = lm_pyramid[l];
        //timer2.reset();
        if (l > 0)
        {
            for (int i = 0; i < (int)quantizers.size(); ++i)
                quantizers[i]->pyrDown(); //all the quantizers aka modality will go down level
        }

        Mat quantized, spread_quantized;
        std::vector<Mat> response_maps;
        for (int i = 0; i < (int)quantizers.size(); ++i)
        {
            quantizers[i]->quantize(quantized); //calculate the quantized image
            //timer2.out("quantize");
            //timer2.reset();
            spread(quantized, spread_quantized,1);
            //timer2.out("spread");
            //timer2.reset();
            computeResponseMaps(spread_quantized, response_maps);
            //timer2.out("computeResponseMaps True");
            //timer2.reset();
            LinearMemories &memories = lm_level[i];
            for (int j = 0; j < 8; ++j)
                linearize(response_maps[j], memories[j], T);
            //timer2.out("linearize ");
            //stimer2.reset();
        }
        string input = "/home/xiangdawei/linemod_python/linemod_cpp_python/result_visual/spread";
        string output_spread_img = Replace(input, "spread", name_i+"spread.png");
        string output_spread = Replace(input, "spread", name_i+"spread.xml");
        string output_quantized_img = Replace(input, "spread", name_i+"quantized.png");
        string output_quantized = Replace(input, "spread", name_i+"quantized.xml");
        string output_response = Replace(input, "spread", name_i+"response.xml");

        //std::cout << "quantized"<< quantized.size()<<"     ";
        //std::cout << "spread_quantized"<< spread_quantized.size()<<"     ";
        //std::cout << "response_maps 0"<< response_maps[0].size()<<"     ";
        //std::cout << "response_maps 7"<< response_maps[7].size()<<"     ";

        FileStorage fs(output_spread, FileStorage::WRITE);
        fs<<"vocabulary"<<spread_quantized;
        fs.release();
        cv::imwrite(output_spread_img, spread_quantized);

        FileStorage fs2(output_quantized, FileStorage::WRITE);
        fs2<<"vocabulary"<<quantized;
        fs2.release();
        cv::imwrite(output_quantized_img, quantized);
        
        save(output_spread, response_maps);

        sizes.push_back(quantized.size());
    }
    
    timer.out("construct response map");

    if (class_ids.empty())
    {
        // Match all templates
        TemplatesMap::const_iterator it = class_templates.begin(), itend = class_templates.end();
        for (; it != itend; ++it)
            matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
    }
    else
    {
        // Match only templates for the requested class IDs
        for (int i = 0; i < (int)class_ids.size(); ++i)
        {
            TemplatesMap::const_iterator it = class_templates.find(class_ids[i]);
            //cout << "template_pyramids size" << it->second.size();
            if (it != class_templates.end())
                matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
        }
    }

    // Sort matches by similarity, and prune any duplicates introduced by pyramid refinement
    std::sort(matches.begin(), matches.end());
    std::vector<Match>::iterator new_end = std::unique(matches.begin(), matches.end());
    matches.erase(new_end, matches.end());

    timer.out("templ match");

    return matches;//, quantized, spread_quantized, response_maps;
}

// Used to filter out weak matches
struct MatchPredicate
{
    MatchPredicate(float _threshold) : threshold(_threshold) {}
    bool operator()(const Match &m) { return m.similarity < threshold; }
    float threshold;
};

void Detector::matchClass(const LinearMemoryPyramid &lm_pyramid,
                          const std::vector<Size> &sizes, //size is the input image size
                          float threshold, std::vector<Match> &matches,
                          const std::string &class_id,
                          const std::vector<TemplatePyramid> &template_pyramids) const
{
#pragma omp declare reduction \
    (omp_insert: std::vector<Match>: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

//#pragma omp parallel for reduction(omp_insert:matches) // compilors parallel operations
    for (size_t template_id = 0; template_id < template_pyramids.size(); ++template_id)
    {
        const TemplatePyramid &tp = template_pyramids[template_id];
        // First match over the whole image at the lowest pyramid level
        /// @todo Factor this out into separate function
        const std::vector<LinearMemories> &lowest_lm = lm_pyramid.back(); //lm_pyramid[1], the  resolution lowest level

        std::vector<Match> candidates;
        {
            // Compute similarity maps for each ColorGradient at lowest pyramid level
            Mat similarities;
            int lowest_start = static_cast<int>(tp.size() - 1);// 2-1 = 1
            //cout <<"lowest_start"<<lowest_start;
            int lowest_T = T_at_level.back(); // back() return the last element, 8
            //cout << "lowest_T" <<lowest_T;
            int num_features = 0;

            {
                const Template &templ = tp[lowest_start]; // 2-nd level templ
                num_features += static_cast<int>(templ.features.size());
                cout << "num_features"<<num_features;
                if (templ.features.size() < 64){
                    similarity_64(lowest_lm[0], templ, similarities, sizes.back(), lowest_T); //0 in lowest_lm[0] means only one quantizer
                    similarities.convertTo(similarities, CV_16U);
                }else if (templ.features.size() < 8192){
                    similarity(lowest_lm[0], templ, similarities, sizes.back(), lowest_T);
                }else{
                    CV_Error(Error::StsBadArg, "feature size too large");
                }
            }
            if (template_id ==61){
                save_single(prefix+"fist_simi.xml",similarities);
            }
            //cout << "first similarity size" << similarities.size();
            // Find initial matches
            //D: similarity shape=[H/T,W/T] ?
            for (int r = 0; r < similarities.rows; ++r)
            {
                ushort *row = similarities.ptr<ushort>(r);
                for (int c = 0; c < similarities.cols; ++c)
                {
                    int raw_score = row[c];
                    //cout<<"raw score"<<raw_score<<"  "<<threshold;
                    float score = (raw_score * 100.f) / (4 * num_features);

                    if (score > threshold)
                    {
                        int offset = lowest_T / 2 + (lowest_T % 2 - 1); //
                        int x = c * lowest_T + offset;
                        int y = r * lowest_T + offset;
                        //cout<<"lowest_T "<<lowest_T<<" c "<<c<< " r "<<r;
                        candidates.push_back(Match(x, y, score, class_id, static_cast<int>(template_id))); //match save the coordinate in the original input image
                    }
                }
            }
            if (template_id ==61){cout<<template_id<<" first level candidate length "<<candidates.size();}
        }


        // Locally refine each match by marching up the pyramid
        for (int l = pyramid_levels - 2; l >= 0; --l) //l=0
        {
            const std::vector<LinearMemories> &lms = lm_pyramid[l];
            int T = T_at_level[l];//4
            //cout << "Second time T"<<T;
            int start = static_cast<int>(l); //0
            Size size = sizes[l];// input image size
            int border = 8 * T;
            int offset = T / 2 + (T % 2 - 1); //
            int max_x = size.width - tp[start].width - border;
            int max_y = size.height - tp[start].height - border;

            Mat similarities2;
            for (int m = 0; m < (int)candidates.size(); ++m)
            {
                Match &match2 = candidates[m];
                int x = match2.x * 2 + 1; /// @todo Support other pyramid distance
                int y = match2.y * 2 + 1;

                // Require 8 (reduced) row/cols to the up/left
                x = std::max(x, border);
                y = std::max(y, border);

                // Require 8 (reduced) row/cols to the down/left, plus the template size
                x = std::min(x, max_x);
                y = std::min(y, max_y);

                // Compute local similarity maps for each ColorGradient
                int numFeatures = 0;

                {
                    const Template &templ = tp[start];
                    numFeatures += static_cast<int>(templ.features.size());

                    if (templ.features.size() < 64){
                        similarityLocal_64(lms[0], templ, similarities2, size, T, Point(x, y));
                        similarities2.convertTo(similarities2, CV_16U);
                    }else if (templ.features.size() < 8192){
                        similarityLocal(lms[0], templ, similarities2, size, T, Point(x, y));
                    }else{
                        CV_Error(Error::StsBadArg, "feature size too large");
                    }
                }

                // Find best local adjustment
                float best_score = 0;
                int best_r = -1, best_c = -1;
                for (int r = 0; r < similarities2.rows; ++r)
                {
                    ushort *row = similarities2.ptr<ushort>(r);
                    for (int c = 0; c < similarities2.cols; ++c)
                    {
                        int score_int = row[c];
                        float score = (score_int * 100.f) / (4 * numFeatures);

                        if (score > best_score)
                        {
                            best_score = score;
                            best_r = r;
                            best_c = c;
                        }
                        if (template_id ==61){
                            cout<<" best_score "<<best_score<<" "<<best_r<<" "<<best_c;
                        }
                    }
                }
                // Update current match
                match2.similarity = best_score;
                cout << "best score" << best_score;
                match2.x = (x / T - 8 + best_c) * T + offset;
                match2.y = (y / T - 8 + best_r) * T + offset;
            }

            // Filter out any matches that drop below the similarity threshold
            std::vector<Match>::iterator new_end = std::remove_if(candidates.begin(), candidates.end(),
                                                                  MatchPredicate(threshold));
            candidates.erase(new_end, candidates.end());
        }

        matches.insert(matches.end(), candidates.begin(), candidates.end());
    }
}

int Detector::addTemplate(const Mat source, const std::string &class_id,
                          const Mat &object_mask, int num_features)
{
    std::vector<TemplatePyramid> &template_pyramids = class_templates[class_id];
    int template_id = static_cast<int>(template_pyramids.size());

    TemplatePyramid tp;
    tp.resize(pyramid_levels);

    {
        // Extract a template at each pyramid level
        Ptr<ColorGradientPyramid> qp = modality->process(source, object_mask);

        if(num_features > 0)
        qp->num_features = num_features;

        for (int l = 0; l < pyramid_levels; ++l)
        {
            /// @todo Could do mask subsampling here instead of in pyrDown()
            if (l > 0)
                qp->pyrDown();

            bool success = qp->extractTemplate(tp[l]);
            if (!success)
                return -1;
        }
    }

    //    Rect bb =
    cropTemplates(tp);

    /// @todo Can probably avoid a copy of tp here with swap
    template_pyramids.push_back(tp);
    return template_id;
}

static cv::Point2f rotate2d(const cv::Point2f inPoint, const double angRad)
{
    cv::Point2f outPoint;
    //CW rotation
    outPoint.x = std::cos(angRad)*inPoint.x - std::sin(angRad)*inPoint.y;
    outPoint.y = std::sin(angRad)*inPoint.x + std::cos(angRad)*inPoint.y;
    return outPoint;
}

static cv::Point2f rotatePoint(const cv::Point2f inPoint, const cv::Point2f center, const double angRad)
{
    return rotate2d(inPoint - center, angRad) + center;  // here rotation is same to get the relative position of inpoint towards center, then do the rotation, and shift to the center.
}

int Detector::addTemplate_rotate(const string &class_id, int zero_id,
                                 float theta, cv::Point2f center)
{
    std::vector<TemplatePyramid> &template_pyramids = class_templates[class_id];
    int template_id = static_cast<int>(template_pyramids.size());

    const auto& to_rotate_tp = template_pyramids[zero_id];  //zero id correspond to "first id", aka the first un-rotated image's template pyramids

    TemplatePyramid tp;
    tp.resize(pyramid_levels);

    for (int l = 0; l < pyramid_levels; ++l)
    {
        if(l>0) center /= 2;

        for(auto& f: to_rotate_tp[l].features){  //here it isn't one but l, means only rotate the feature point in the first template
            Point2f p;
            p.x = f.x + to_rotate_tp[l].tl_x;
            p.y = f.y + to_rotate_tp[l].tl_y;
            Point2f p_rot = rotatePoint(p, center, -theta/180*CV_PI);

            Feature f_new;
            f_new.x = int(p_rot.x + 0.5f);
            f_new.y = int(p_rot.y + 0.5f);

            f_new.theta = f.theta - theta;
            while(f_new.theta > 360) f_new.theta -= 360;
            while(f_new.theta < 0) f_new.theta += 360;

            f_new.label = int(f_new.theta * 16 / 360 + 0.5f);
            f_new.label &= 7;


            tp[l].features.push_back(f_new);
        }
        tp[l].pyramid_level = l;
    }

    cropTemplates(tp);

    template_pyramids.push_back(tp);
    return template_id;
}
const std::vector<Template> &Detector::getTemplates(const std::string &class_id, int template_id) const
{
    TemplatesMap::const_iterator i = class_templates.find(class_id);
    CV_Assert(i != class_templates.end());
    CV_Assert(i->second.size() > size_t(template_id));
    return i->second[template_id];
}

int Detector::numTemplates() const
{
    int ret = 0;
    TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
    for (; i != iend; ++i)
        ret += static_cast<int>(i->second.size());
    return ret;
}

int Detector::numTemplates(const std::string &class_id) const
{
    TemplatesMap::const_iterator i = class_templates.find(class_id);
    if (i == class_templates.end())
        return 0;
    return static_cast<int>(i->second.size());
}

std::vector<std::string> Detector::classIds() const
{
    std::vector<std::string> ids;
    TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
    for (; i != iend; ++i)
    {
        ids.push_back(i->first);
    }

    return ids;
}

void Detector::read(const FileNode &fn)
{
    class_templates.clear();
    pyramid_levels = fn["pyramid_levels"];
    fn["T"] >> T_at_level;

    modality = makePtr<ColorGradient>();
}

void Detector::write(FileStorage &fs) const
{
    fs << "pyramid_levels" << pyramid_levels;
    fs << "T" << T_at_level;

    modality->write(fs);
}

std::string Detector::readClass(const FileNode &fn, const std::string &class_id_override)
{
    // Detector should not already have this class
    String class_id;
    if (class_id_override.empty())
    {
        String class_id_tmp = fn["class_id"];
        CV_Assert(class_templates.find(class_id_tmp) == class_templates.end());
        class_id = class_id_tmp;
    }
    else
    {
        class_id = class_id_override;
    }

    TemplatesMap::value_type v(class_id, std::vector<TemplatePyramid>());
    std::vector<TemplatePyramid> &tps = v.second;
    int expected_id = 0;

    FileNode tps_fn = fn["template_pyramids"];
    tps.resize(tps_fn.size());
    FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
    for (; tps_it != tps_it_end; ++tps_it, ++expected_id)
    {
        int template_id = (*tps_it)["template_id"];
        CV_Assert(template_id == expected_id);
        FileNode templates_fn = (*tps_it)["templates"];
        tps[template_id].resize(templates_fn.size());

        FileNodeIterator templ_it = templates_fn.begin(), templ_it_end = templates_fn.end();
        int idx = 0;
        for (; templ_it != templ_it_end; ++templ_it)
        {
            tps[template_id][idx++].read(*templ_it);
        }
    }

    class_templates.insert(v);
    return class_id;
}

void Detector::writeClass(const std::string &class_id, FileStorage &fs) const
{
    TemplatesMap::const_iterator it = class_templates.find(class_id);
    CV_Assert(it != class_templates.end());
    const std::vector<TemplatePyramid> &tps = it->second;

    fs << "class_id" << it->first;
    fs << "pyramid_levels" << pyramid_levels;
    fs << "template_pyramids"
       << "[";
    for (size_t i = 0; i < tps.size(); ++i)
    {
        const TemplatePyramid &tp = tps[i];
        fs << "{";
        fs << "template_id" << int(i); //TODO is this cast correct? won't be good if rolls over...
        fs << "templates"
           << "[";
        for (size_t j = 0; j < tp.size(); ++j)
        {
            fs << "{";
            tp[j].write(fs);
            fs << "}"; // current template
        }
        fs << "]"; // templates
        fs << "}"; // current pyramid
    }
    fs << "]"; // pyramids
}

void Detector::readClasses(const std::vector<std::string> &class_ids,
                           const std::string &format)
{
    for (size_t i = 0; i < class_ids.size(); ++i)
    {
        const String &class_id = class_ids[i];
        String filename = cv::format(format.c_str(), class_id.c_str());
        FileStorage fs(filename, FileStorage::READ);
        readClass(fs.root());
    }
}

void Detector::writeClasses(const std::string &format) const
{
    TemplatesMap::const_iterator it = class_templates.begin(), it_end = class_templates.end();
    for (; it != it_end; ++it)
    {
        const String &class_id = it->first;
        String filename = cv::format(format.c_str(), class_id.c_str());
        FileStorage fs(filename, FileStorage::WRITE);
        writeClass(class_id, fs);
    }
}

} // namespace line2Dup











#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>
using namespace std;
using namespace cv;

static std::string prefix = "/home/xiangdawei/linemod_python/linemod_cpp_python/test_files/";


// NMS, got from cv::dnn so we don't need opencv contrib
// just collapse it
namespace  cv_dnn {
namespace
{

template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}

} // namespace

inline void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                      std::vector<std::pair<float, int> >& score_index_vec)
{
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > threshold)
        {
            score_index_vec.push_back(std::make_pair(scores[i], i));
        }
    }
    std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                     SortScorePairDescend<int>);
    if (top_k > 0 && top_k < (int)score_index_vec.size())
    {
        score_index_vec.resize(top_k);
    }
}

template <typename BoxType>
inline void NMSFast_(const std::vector<BoxType>& bboxes,
      const std::vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      std::vector<int>& indices, float (*computeOverlap)(const BoxType&, const BoxType&))
{
    CV_Assert(bboxes.size() == scores.size());
    std::vector<std::pair<float, int> > score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices.clear();
    for (size_t i = 0; i < score_index_vec.size(); ++i) {
        const int idx = score_index_vec[i].second;
        bool keep = true;
        for (int k = 0; k < (int)indices.size() && keep; ++k) {
            const int kept_idx = indices[k];
            float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
            keep = overlap <= adaptive_threshold;
        }
        if (keep)
            indices.push_back(idx);
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
          adaptive_threshold *= eta;
        }
    }
}


// copied from opencv 3.4, not exist in 3.0
template<typename _Tp> static inline
double jaccardDistance__(const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
    _Tp Aa = a.area();
    _Tp Ab = b.area();

    if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
        // jaccard_index = 1 -> distance = 0
        return 0.0;
    }

    double Aab = (a & b).area();
    // distance = 1 - jaccard_index
    return 1.0 - Aab / (Aa + Ab - Aab);
}

template <typename T>
static inline float rectOverlap(const T& a, const T& b)
{
    return 1.f - static_cast<float>(jaccardDistance__(a, b));
}

void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
                          const float score_threshold, const float nms_threshold,
                          std::vector<int>& indices, const float eta=1, const int top_k=0)
{
    NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
}

}

void scale_test(string mode = "test"){
    int num_feature = 150;

    // feature numbers(how many ori in one templates?)
    // two pyramids, lower pyramid(more pixels) in stride 4, lower in stride 8
    line2Dup::Detector detector(num_feature, {4, 8});

//    mode = "test";
    if(mode == "train"){
        Mat img = cv::imread(prefix+"case0/templ/circle.png");
        assert(!img.empty() && "check your img path");
        shape_based_matching::shapeInfo_producer shapes(img);

        shapes.scale_range = {0.1f, 1};
        shapes.scale_step = 0.01f;
        shapes.produce_infos();

        std::vector<shape_based_matching::Info> infos_have_templ;
        string class_id = "circle";
        for(auto& info: shapes.infos){

            // template img, id, mask,
            //feature numbers(missing it means using the detector initial num)
            int templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info),
                                                int(num_feature*info.scale));
            std::cout << "templ_id: " << templ_id << std::endl;

            // may fail when asking for too many feature_nums for small training img
            if(templ_id != -1){  // only record info when we successfully add template
                infos_have_templ.push_back(info);
            }
        }

        // save templates
        detector.writeClasses(prefix+"case0/%s_templ.yaml");

        // save infos,
        // in this simple case infos are not used
        shapes.save_infos(infos_have_templ, prefix + "case0/circle_info.yaml");
        std::cout << "train end" << std::endl << std::endl;

    }else if(mode=="test"){
        std::vector<std::string> ids;

        // read templates
        ids.push_back("circle");
        detector.readClasses(ids, prefix+"case0/%s_templ.yaml");

        Mat test_img = imread(prefix+"case0/1.jpg");
        assert(!test_img.empty() && "check your img path");

        // make the img having 32*n width & height
        // at least 16*n here for two pyrimads with strides 4 8
        int stride = 32;
        int n = test_img.rows/stride;
        int m = test_img.cols/stride;
        Rect roi(0, 0, stride*m , stride*n);
        Mat img = test_img(roi).clone();
        assert(img.isContinuous());

        Timer timer;
        // match, img, min socre, ids
        auto matches = detector.match(img, 90, "test", ids);
        // one output match:
        // x: top left x
        // y: top left y
        // template_id: used to find templates
        // similarity: scores, 100 is best
        timer.out();

        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 5;
        if(top5>matches.size()) top5=matches.size();
        for(size_t i=0; i<top5; i++){
            auto match = matches[i];
            auto templ = detector.getTemplates("circle",
                                               match.template_id);
            // template:
            // nums: num_pyramids * num_modality (modality, depth or RGB, always 1 here)
            // template[0]: lowest pyrimad(more pixels)
            // template[0].width: actual width of the matched template
            // template[0].tl_x / tl_y: topleft corner when cropping templ during training
            // In this case, we can regard width/2 = radius
            int x =  templ[0].width/2 + match.x;
            int y = templ[0].height/2 + match.y;
            int r = templ[0].width/2;
            Scalar color(255, rand()%255, rand()%255);

            cv::putText(img, to_string(int(round(match.similarity))),
                        Point(match.x+r-10, match.y-3), FONT_HERSHEY_PLAIN, 2, color);
            cv::circle(img, {x, y}, r, color, 2);
        }

        imshow("img", img);
        waitKey(0);

        std::cout << "test end" << std::endl << std::endl;
    }
}

void angle_test(string mode = "test", bool use_rot = true){
    line2Dup::Detector detector(128, {4, 8});

    if(mode != "test"){
        Mat img = imread(prefix+"case1/train.png");
        
        assert(!img.empty() && "check your img path");

        Rect roi(130, 110, 270, 270);
        img = img(roi).clone();
        Mat mask = Mat(img.size(), CV_8UC1, {255});

        // padding to avoid rotating out
        int padding = 100;
        cv::Mat padded_img = cv::Mat(img.rows + 2*padding, img.cols + 2*padding, img.type(), cv::Scalar::all(0));
        img.copyTo(padded_img(Rect(padding, padding, img.cols, img.rows)));

        cv::Mat padded_mask = cv::Mat(mask.rows + 2*padding, mask.cols + 2*padding, mask.type(), cv::Scalar::all(0));
        mask.copyTo(padded_mask(Rect(padding, padding, img.cols, img.rows)));

        shape_based_matching::shapeInfo_producer shapes(padded_img, padded_mask);
        shapes.angle_range = {0, 360};
        shapes.angle_step = 1;

        shapes.scale_range = {1}; // support just one
        shapes.produce_infos();
        std::vector<shape_based_matching::Info> infos_have_templ;
        string class_id = "test";

        bool is_first = true;

        // for other scales you want to re-extract points: 
        // set shapes.scale_range then produce_infos; set is_first = false;

        int first_id = 0;
        float first_angle = 0;
        for(auto& info: shapes.infos){
            Mat to_show = shapes.src_of(info);

            std::cout << "\ninfo.angle: " << info.angle << std::endl;
            int templ_id;

            if(is_first){
                templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info));
                first_id = templ_id;
                first_angle = info.angle;

                if(use_rot) is_first = false;
            }else{
                templ_id = detector.addTemplate_rotate(class_id, first_id,
                                                       info.angle-first_angle,
                                                {shapes.src.cols/2.0f, shapes.src.rows/2.0f});
            }

            auto templ = detector.getTemplates("test", templ_id);
            for(int i=0; i<templ[0].features.size(); i++){
                auto feat = templ[0].features[i];
                cv::circle(to_show, {feat.x+templ[0].tl_x, feat.y+templ[0].tl_y}, 3, {0, 0, 255}, -1);
            }
            
            // will be faster if not showing this
            imshow("train", to_show);
            waitKey(1);

            std::cout << "templ_id: " << templ_id << std::endl;
            if(templ_id != -1){
                infos_have_templ.push_back(info);
            }
        }
        detector.writeClasses(prefix+"case1/%s_templ.yaml");
        shapes.save_infos(infos_have_templ, prefix + "case1/test_info.yaml");
        std::cout << "train end" << std::endl << std::endl;
    }else if(mode=="test"){
        std::vector<std::string> ids;
        ids.push_back("test");
        detector.readClasses(ids, prefix+"case1/%s_templ.yaml");
        std::cout << "read  yaml";
        // angle & scale are saved here, fetched by match id
        shape_based_matching::shapeInfo_producer producer;
        auto infos = producer.load_infos(prefix + "case1/test_info.yaml");

        Mat test_img = imread(prefix+"case1/test.png");
        std::cout << "read  test image";
        assert(!test_img.empty() && "check your img path");

        int padding = 250;
        cv::Mat padded_img = cv::Mat(test_img.rows + 2*padding,
                                     test_img.cols + 2*padding, test_img.type(), cv::Scalar::all(0));
        test_img.copyTo(padded_img(Rect(padding, padding, test_img.cols, test_img.rows)));

        int stride = 16;
        int n = padded_img.rows/stride;
        int m = padded_img.cols/stride;
        Rect roi(0, 0, stride*m , stride*n);
        Mat img = padded_img(roi).clone();
        assert(img.isContinuous());

//        cvtColor(img, img, CV_BGR2GRAY);

        std::cout << "test img size: " << img.rows * img.cols << std::endl << std::endl;

        Timer timer;
        auto matches = detector.match(img, 90, "test", ids);
        timer.out();

        if(img.channels() == 1) cvtColor(img, img, CV_GRAY2BGR);

        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 1;
        if(top5>matches.size()) top5=matches.size();
        for(size_t i=0; i<top5; i++){
            auto match = matches[i];
            auto templ = detector.getTemplates("test",
                                               match.template_id);

            // 270 is width of template image
            // 100 is padding when training
            // tl_x/y: template croping topleft corner when training

            float r_scaled = 270/2.0f*infos[match.template_id].scale;

            // scaling won't affect this, because it has been determined by warpAffine
            // cv::warpAffine(src, dst, rot_mat, src.size()); last param
            float train_img_half_width = 270/2.0f + 100;
            float train_img_half_height = 270/2.0f + 100;

            // center x,y of train_img in test img
            float x =  match.x - templ[0].tl_x + train_img_half_width;
            float y =  match.y - templ[0].tl_y + train_img_half_height;

            cv::Vec3b randColor;
            randColor[0] = rand()%155 + 100;
            randColor[1] = rand()%155 + 100;
            randColor[2] = rand()%155 + 100;
            for(int i=0; i<templ[0].features.size(); i++){
                auto feat = templ[0].features[i];
                cv::circle(img, {feat.x+match.x, feat.y+match.y}, 3, randColor, -1);
            }

            cv::putText(img, to_string(int(round(match.similarity))),
                        Point(match.x+r_scaled-10, match.y-3), FONT_HERSHEY_PLAIN, 2, randColor);

            cv::RotatedRect rotatedRectangle({x, y}, {2*r_scaled, 2*r_scaled}, -infos[match.template_id].angle);

            cv::Point2f vertices[4];
            rotatedRectangle.points(vertices);
            for(int i=0; i<4; i++){
                int next = (i+1==4) ? 0 : (i+1);
                cv::line(img, vertices[i], vertices[next], randColor, 2);
            }

            std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
            std::cout << "match.similarity: " << match.similarity << std::endl;
        }

        imshow("img", img);
        waitKey(0);

        std::cout << "test end" << std::endl << std::endl;
    }
}

void noise_test(string mode = "test"){
    line2Dup::Detector detector(30, {4, 8});

    if(mode == "train"){
        Mat img = imread(prefix+"case2/train.png");
        assert(!img.empty() && "check your img path");
        Mat mask = Mat(img.size(), CV_8UC1, {255});

        shape_based_matching::shapeInfo_producer shapes(img, mask);
        shapes.angle_range = {0, 360};
        shapes.angle_step = 1;
        shapes.produce_infos();
        std::vector<shape_based_matching::Info> infos_have_templ;
        string class_id = "test";
        for(auto& info: shapes.infos){
            imshow("train", shapes.src_of(info));
            waitKey(1);

            std::cout << "\ninfo.angle: " << info.angle << std::endl;
            int templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info));
            std::cout << "templ_id: " << templ_id << std::endl;
            if(templ_id != -1){
                infos_have_templ.push_back(info);
            }
        }
        detector.writeClasses(prefix+"case2/%s_templ.yaml");
        shapes.save_infos(infos_have_templ, prefix + "case2/test_info.yaml");
        std::cout << "train end" << std::endl << std::endl;
    }else if(mode=="test"){
        std::vector<std::string> ids;
        ids.push_back("test");
        detector.readClasses(ids, prefix+"case2/%s_templ.yaml");

        Mat test_img = imread(prefix+"case2/test.png");
        assert(!test_img.empty() && "check your img path");

        // cvtColor(test_img, test_img, CV_BGR2GRAY);

        int stride = 16;
        int n = test_img.rows/stride;
        int m = test_img.cols/stride;
        Rect roi(0, 0, stride*m , stride*n);

        test_img = test_img(roi).clone();

        Timer timer;
        auto matches = detector.match(test_img, 90, "test",  ids);
        timer.out();

        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 500;
        if(top5>matches.size()) top5=matches.size();

        vector<Rect> boxes;
        vector<float> scores;
        vector<int> idxs;
        for(auto match: matches){
            Rect box;
            box.x = match.x;
            box.y = match.y;

            auto templ = detector.getTemplates("test",
                                               match.template_id);

            box.width = templ[0].width;
            box.height = templ[0].height;
            boxes.push_back(box);
            scores.push_back(match.similarity);
        }
        cv_dnn::NMSBoxes(boxes, scores, 0, 0.5f, idxs);

        for(auto idx: idxs){
            auto match = matches[idx];
            auto templ = detector.getTemplates("test",
                                               match.template_id);

            int x =  templ[0].width + match.x;
            int y = templ[0].height + match.y;
            int r = templ[0].width/2;
            cv::Vec3b randColor;
            randColor[0] = rand()%155 + 100;
            randColor[1] = rand()%155 + 100;
            randColor[2] = rand()%155 + 100;

            for(int i=0; i<templ[0].features.size(); i++){
                auto feat = templ[0].features[i];
                cv::circle(test_img, {feat.x+match.x, feat.y+match.y}, 2, randColor, -1);
            }

            cv::putText(test_img, to_string(int(round(match.similarity))),
                        Point(match.x+r-10, match.y-3), FONT_HERSHEY_PLAIN, 2, randColor);
            cv::rectangle(test_img, {match.x, match.y}, {x, y}, randColor, 2);

            std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
            std::cout << "match.similarity: " << match.similarity << std::endl;
        }

        imshow("img", test_img);
        waitKey(0);

        std::cout << "test end" << std::endl << std::endl;
    }
}

void MIPP_test(){
    std::cout << "MIPP tests" << std::endl;
    std::cout << "----------" << std::endl << std::endl;

    std::cout << "Instr. type:       " << mipp::InstructionType                  << std::endl;
    std::cout << "Instr. full type:  " << mipp::InstructionFullType              << std::endl;
    std::cout << "Instr. version:    " << mipp::InstructionVersion               << std::endl;
    std::cout << "Instr. size:       " << mipp::RegisterSizeBit       << " bits" << std::endl;
    std::cout << "Instr. lanes:      " << mipp::Lanes                            << std::endl;
    std::cout << "64-bit support:    " << (mipp::Support64Bit    ? "yes" : "no") << std::endl;
    std::cout << "Byte/word support: " << (mipp::SupportByteWord ? "yes" : "no") << std::endl;

#ifndef has_max_int8_t
        std::cout << "in this SIMD, int8 max is not inplemented by MIPP" << std::endl;
#endif

#ifndef has_shuff_int8_t
        std::cout << "in this SIMD, int8 shuff is not inplemented by MIPP" << std::endl;
#endif

    std::cout << "----------" << std::endl << std::endl;
}

int main(){
    // scale_test("test");
    MIPP_test();
    angle_test("test", false); // test or train
    // noise_test("test");
    
    return 0;
}
