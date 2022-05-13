// main.cpp
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include "opencv2/imgcodecs.hpp"

using namespace cv;
using namespace std;

struct BGR {
    uchar blue;
    uchar green;
    uchar red;
};

Mat readImg(String img_path, int flag = IMREAD_COLOR){
    Mat img = imread(img_path, flag);
    if (img.empty())
    {
        std::cout << "!!! Failed imread(): image not found" << std::endl;
        // don't let the execution continue, else imshow() will crash.
    }
    return img;
}

Mat cannyImg(Mat img){
    Mat src1 = img;
    Mat out;
    Mat dst, edge, gray;

    //创建一个与src1一样的矩阵
    dst.create(src1.size(), src1.type());

    //将原图像转为灰度
    cvtColor(src1, gray, COLOR_RGB2GRAY);

    //滤波(降噪)
    blur(gray, edge, Size(3, 3));

    //canny
    Mat out2;
    Canny(edge, out2, 15, 10);
    dst = Scalar::all(0);
    src1.copyTo(dst, out2);
    return out2;
}

Mat canny2threshold(Mat src) // Binary threshold TODO
{
    Mat dst;
    double thresh = 100;
    int maxVal = 255;
    threshold(src, dst, thresh, maxVal, THRESH_BINARY);
    return dst;
}

//Mat gradient_test(Mat input){
//    if (input.size().width <= 1) return input;
//    vector<double> res;
//    for(int j=0; j<input.size(); j++) {
//        int j_left = j - 1;
//        int j_right = j + 1;
//        if (j_left < 0) {
//            j_left = 0; // use your own boundary handler
//            j_right = 1;
//        }
//        if (j_right >= input.size()){
//            j_right = input.size() - 1;
//            j_left = j_right - 1;
//        }
//        // gradient value at position j
//        double dist_grad = (input[j_right] - input[j_left]) / 2.0;
//        res.push_back(dist_grad);
//    }
//    return res;
//}
//
std::tuple<Mat, Mat> gradient_minus_mean_2d(Mat input, float myMAtMean){
    // TODO minus mean
    int height = input.rows;
    int width = input.cols;
    if(height <= 1 && width <= 1) return std::make_tuple(input,input);
    Mat res_x = Mat::zeros(height, width, CV_32F);;
    Mat res_y = Mat::zeros(height, width, CV_32F);;


    for (int i = 0; i < input.rows; i++)
    {

        float* res_x_pointer = res_x.ptr<float>(i);
//        float* input_p = input.ptr<float>(i);//get the address of row i;
        // Loop over all columns and compute the horizontal gradient
        for ( int j = 0; j < input.cols; j++)
        {
            int pixel_right = 0;
            int pixel_left = 0;
            int j_left = j - 1;
            int j_right = j + 1;
            if (j_left < 0) {
                j_left = 0; // use your own boundary handler
                j_right = 1;
            }
            if (j_right >= input.size().width){
                j_right = input.size().width - 1;
                j_left = j_right - 1;
            }
//            pixel_right = input.at<float>(j_right,j);
//            pixel_left = input.at<float>(j_left,j);
            Scalar intensity_p_right=input.at<uchar>(i,j_right);
            Scalar intensity_p_left=input.at<uchar>(i,j_left);
            pixel_right = intensity_p_right[0];
            pixel_left = intensity_p_left[0];
//            pixel_right = input_p[j];
//            pixel_left = input_p[j];
//            if(pixel_left != 0 && pixel_right != 0){
//                cout << "pixel_right=" << pixel_right << ",pixel_left=" << pixel_left << endl;
//            }
            float dist_grad = 0;
            if((pixel_right == 0 && pixel_left == 0) || (pixel_left == pixel_left)){
                dist_grad = -myMAtMean;
            }else{
                dist_grad = (pixel_right - pixel_left) / 2.0 - myMAtMean;
            }
            res_x_pointer[j] = dist_grad;
        }
    }
    // Do the y direction gradient computation
    for (int i = 0; i < input.rows; i++)  //
    {
        float* res_y_pointer = res_y.ptr<float>(i);
        int i_below = i + 1; // y direction
        int i_upside = i - 1;
        if (i_upside < 0) {
            i_below = 1; // use your own boundary handler
            i_upside = 0;
        }
        if (i_below >= input.size().height){
            i_below = input.size().height - 1;
            i_upside = i_below - 1;
        }

//        float* input_p_below = input.ptr<float>(i_below);//get the address of row i;
//        float* input_p_upside = input.ptr<float>(i_upside);//get the address of row i;
        // Loop over all columns and compute the horizontal gradient
        for ( int j = 0; j < input.cols; j++)
        {
            int pixel_upside = 0;
            int pixel_below = 0;
//            pixel_int = input.at<float>(i, j);
//            pixel_upside = input.at<float>(i_upside,j);
//            pixel_below = input.at<float>(i_below,j);
            Scalar intensity_p_below=input.at<uchar>(i_below,j);
            Scalar intensity_p_upside=input.at<uchar>(i_upside,j);
            pixel_upside = intensity_p_upside[0];
            pixel_below = intensity_p_below[0];

//            pixel_upside = input_p_upside[j];
//            pixel_below = input_p_below[j];

            float dist_grad = 0;
            if((pixel_upside == 0 && pixel_below == 0) || (pixel_upside == pixel_below)){
                dist_grad = -myMAtMean;
            }else{
                dist_grad = (pixel_below - pixel_upside) / 2.0 - myMAtMean;
            }
//            res_y.push_back(dist_grad);
            res_y_pointer[j] = dist_grad;

        }
    }
    return std::make_tuple(res_x,res_y);
}
Mat canny2points_3channels(Mat img){
    int img_x = img.rows;
    int img_y = img.cols;
    int gridsize = ceil(img_x/500); // TODO how to define gridsize
    int count = 0;
    if(gridsize < 3)
        gridsize = 3;
    double k = 0.04;
    Mat points_map(img.size(), CV_32FC3, cv::Scalar(255,255,255)); // greyscale
//    Mat points_map = Mat::zeros(img.size(), CV_8UC3); // 3-channels
//    Mat harris_response = Mat::zeros(img.size(), CV_32F);
    for(int i = 0; i < img_x; ++i){
        if(i % gridsize != 0)
            continue;
        float* points_map_pointer = points_map.ptr<float>(i);

        for(int j = 0; j < img_y; ++j){
            if(j % gridsize != 0)
                continue;

            // TODO create harris points

            // Crop a matrix from source img
            Mat crop_img = img(Range(i, i+gridsize), Range(j, j+gridsize));
            Scalar tempVal = mean( crop_img );
            float myMAtMean = tempVal.val[0];
            if(myMAtMean == 0 )continue;

            // Gridsize square gradient minus matrix mean
            Mat block_x,block_y;
            std::tie(block_x,block_y) = gradient_minus_mean_2d(crop_img, myMAtMean);
            std::tuple<Mat, Mat> mat_tuple = std::tie(block_x,block_y);

            // Calculate the covariance matrix
            Mat ixx = cv::Mat(block_x.size(), CV_32F);
            Mat iyy = cv::Mat(block_x.size(), CV_32F);
            Mat ixy = cv::Mat(block_x.size(), CV_32F);
            ixx = block_x * block_x;
            iyy = block_y * block_y;
            ixy = block_x * block_y;
//            cout << "ixx = " << ixx << endl;
//            cout << "iyy = " << iyy << endl;
//            cout << "ixy = " << ixy << endl;
            float sxx = sum(ixx)[0];
            float syy = sum(iyy)[0];
            float sxy = sum(ixy)[0];
//            cout << "sxx = " << sxx << endl;
//            cout << "syy = " << syy << endl;
//            cout << "sxy = " << sxy << endl;
            Mat covariance_mat = cv::Mat(2, 2, CV_32F, {sxx, sxy, sxy, syy});
            Mat E, V;

            eigen(covariance_mat,E,V);
            // Calculate the Harris Response
            cout << "E = " << E << endl;
            cout << "V = " << V << endl;
            double minVal, maxVal;
            minMaxLoc(V, &minVal, &maxVal);

//            points_map_pointer[j] = 0;

//            points_map.at<float>(i,j) = 0;
//            cout << points_map.at<float>(i,j) << endl;
            // TODO Use thresholding on eigenvalues to detect the Harris corner points.
            if(maxVal > 0){
//                cout << "eig_min > 0!" << endl;
//                Vec3b & color = points_map.at<Vec3b>(i,j);
//
//                // ... do something to the color ....
//                color[0] = 0;
//                color[1] = 0;
//                color[2] = 255;
//
//                // set pixel
//                points_map.at<Vec3b>(i,j) = color;
                points_map.at<cv::Vec3b>(i,j)[0]=0; // change it to white
                points_map.at<cv::Vec3b>(i,j)[1]=0;
                points_map.at<cv::Vec3b>(i,j)[2]=255;
                cout << "BGR = " << endl;
                cout << points_map.at<cv::Vec3b>(i,j)[0] <<
                points_map.at<cv::Vec3b>(i,j)[1] <<
                points_map.at<cv::Vec3b>(i,j)[2] ;

//                points_map_pointer[j] = {255,255,255};
            }
        }
    }
    return points_map;
}

Mat canny2points(Mat img){
    int img_x = img.rows;
    int img_y = img.cols;
    int gridsize = ceil(img_x/500); // TODO how to define grid size
    if(gridsize < 3)
        gridsize = 3;
    double k = 0.04;
    int count = 0;

    Mat points_map(img.size(), CV_32FC1, cv::Scalar(0)); // greyscale
//    Mat points_map = Mat::zeros(img.size(), CV_8UC3); // 3-channels
//    Mat harris_response = Mat::zeros(img.size(), CV_32F);
    for(int i = 0; i < img_x; ++i){
        if(i % gridsize != 0)
            continue;
        float* points_map_pointer = points_map.ptr<float>(i);

        for(int j = 0; j < img_y; ++j){
            if(j % gridsize != 0)
                continue;

            // TODO create harris points

            // Crop a matrix from source img
            Mat crop_img = img(Range(i, i+gridsize), Range(j, j+gridsize));

//            Scalar tempVal = mean( crop_img );
//            float myMatMean = tempVal.val[0];
//            if(myMatMean == 0 )continue;
            float sum_mean = 0;
            for(int i = 0; i < crop_img.rows; i++){
//                uchar* img_p = img.ptr<uchar>(i);

                for(int j = 0; j < crop_img.cols; j++){
                    Scalar intensity=crop_img.at<uchar>(i,j);
                    sum_mean += intensity[0];
//                    if(intensity[0] != 0){
//                        cout << "pixel value of (" << i << "," << j << ") = " << intensity[0] << ";" << endl;
//                    }
                }
            }
            float myMatMean = 0;
            myMatMean = sum_mean / (crop_img.cols * crop_img.rows);
            if(myMatMean == 0 ){
//                points_map_pointer[j] = 255;
                continue;
            }
            // Gridsize square gradient minus matrix mean
            Mat block_x,block_y;
            std::tie(block_x,block_y) = gradient_minus_mean_2d(crop_img, myMatMean);
            std::tuple<Mat, Mat> mat_tuple = std::tie(block_x,block_y);

            // Calculate the covariance matrix
            Mat ixx = cv::Mat(block_x.size(), CV_32F);
            Mat iyy = cv::Mat(block_x.size(), CV_32F);
            Mat ixy = cv::Mat(block_x.size(), CV_32F);
            ixx = block_x * block_x;
            iyy = block_y * block_y;
            ixy = block_x * block_y;
//            cout << "ixx = " << ixx << endl;
//            cout << "iyy = " << iyy << endl;
//            cout << "ixy = " << ixy << endl;
            float sxx = sum(ixx)[0];
            float syy = sum(iyy)[0];
            float sxy = sum(ixy)[0];

            float temp[2][2] = {{sxx, sxy}, {sxy, syy}};
            Mat covariance_mat = (Mat_<float>(2,2) << sxx, sxy, sxy, syy);

//            Mat covariance_mat = cv::Mat(2, 2, CV_32F);
//            covariance_mat.data = temp[0];
            Mat E, V;

            eigen(covariance_mat,E,V);
            // Calculate the Harris Response
//            cout << "myMatMean = " << myMatMean << endl;
//            cout << "block_x = " << block_x << endl;
//            cout << "block_y = " << block_y << endl;
//            cout << "sxx = " << sxx << endl;
//            cout << "syy = " << syy << endl;
//            cout << "sxy = " << sxy << endl;
//            cout << "covariance_mat = " << covariance_mat << endl;
//            cout << "E = " << E << endl;
//            cout << "V = " << V << endl;
            double minVal, maxVal;
            minMaxLoc(E, &minVal, &maxVal);
//            cout << maxVal << endl;
//            cout << maxVal << endl;

//            points_map_pointer[j] = 0;

//            points_map.at<float>(i,j) = 0;
//            cout << points_map.at<float>(i,j) << endl;

            // TODO Use thresholding on eigenvalues to detect the Harris corner points.
            if(minVal > 0){
                count ++;
//                cout << "eig_min > 0!" << endl;
                points_map_pointer[j] = 255;
            }
        }
    }
    cout << "count = " << count << endl;
    return points_map;
}

Mat points2adjusted(Mat points_map){
    int img_x = points_map.cols;
    int img_y = points_map.rows;
    int grid_size = ceil(img_x/50); // TODO how to define gridsize
    int count = 0;
    if(grid_size < 1)
        grid_size = 1;
    Mat kernel_ellipse = cv::getStructuringElement(cv::MORPH_ELLIPSE, {2*grid_size, 2*grid_size});
//    Mat kernel_cross = cv::getStructuringElement(cv::MORPH_CROSS, {2*grid_size, 2*grid_size});
//    Mat kernel_rect = cv::getStructuringElement(cv::MORPH_RECT, {2*grid_size, 2*grid_size});
    Mat points_map_dilated(points_map.size(), CV_32FC1, cv::Scalar(0)); // greyscale
    cv::dilate(points_map, points_map_dilated, kernel_ellipse);
    return points_map_dilated;

}


// Draw contours using flood fill
// TODO paint contour
Mat drawContours(Mat adjusted_img, Mat src_img){
    cv::Vec3b white(255, 255, 255);
    cv::Vec3b black(0, 0, 0);
    Mat mask;
    vector<vector<Point> > contours;
    cv::cvtColor(src_img, src_img, cv::COLOR_BGR2GRAY);
//    cv::threshold(src_img, mask, 125, 255, THRESH_BINARY_INV);
    bitwise_not(src_img,src_img);//颜色反转
//    cout << src_img.channels() << endl;
    for(int i = 0; i < adjusted_img.rows; i++){
        cv::Vec3b* row = adjusted_img.ptr<cv::Vec3b>(i);
        cv::Vec3b* row_src = src_img.ptr<cv::Vec3b>(i);

        for(int j = 0; j < adjusted_img.cols; j++){
            cout << "row[j] = " << row[j] << ", row_src[j] = " << row_src[j] << endl;
//            if(row[j] == black && row_src[j] == white)
            if(row[j] == black)
            {
//                cv::floodFill(src_img, cv::Point(i,j), cv::Scalar(255,0,0), (cv::Rect*)0, cv::Scalar(), cv::Scalar(200,255,255), cv::FLOODFILL_FIXED_RANGE);
//                cv::floodFill(src_img, cv::Point(i,j), cv::Scalar(255,0,0), (cv::Rect*)0, cv::Scalar(50,50,50), cv::Scalar(50,50,50), cv::FLOODFILL_FIXED_RANGE);
//                  cv::floodFill(adjusted_img, src_img, cv::Point(i,j), cv::Scalar(255,0,0), (cv::Rect*)0, cv::Scalar(), cv::Scalar(), cv::FLOODFILL_MASK_ONLY);
                // TODO collecting
            }else{
                continue;
            }
//            Scalar intensity=adjusted_img.at<uchar>(i,j);
//            int x = adjusted_img.at<Vec3b>(i, j)[0];
//            int y = adjusted_img.at<Vec3b>(i, j)[1];
//            int z = adjusted_img.at<Vec3b>(i, j)[2];
//            int src_x = src_img.at<Vec3b>(i, j)[0];
//            int src_y = src_img.at<Vec3b>(i, j)[1];
//            int src_z = src_img.at<Vec3b>(i, j)[2];
//            cout << "bgr_blue:" << src_x << ",bgr_green:" << src_y << ",bgr_red:" << src_z << endl;

//            if((x == 0 && y == 0 && z == 0) && (src_x == 255 && src_y == 255 && src_z == 255)){ // if pixel color is black
//                floodFill(src_img, Point(i,j), Scalar(0, 0,255), (cv::Rect*)0, Scalar (0,0,0), Scalar (255,255,0), cv::FLOODFILL_FIXED_RANGE );
//                floodFill(adjusted_img,src_img ,Point(i,j), Scalar(100, 100,100), (cv::Rect*)0, Scalar (160,160,160), Scalar (160,160,160), cv::FLOODFILL_MASK_ONLY );

                //                intensity=src_img.at<uchar>(i,j);
//                Vec3b & color = src_img.at<Vec3b>(i,j);
//                BGR& bgr = src_img.ptr<BGR>(i)[j];
//                cout << "bgr_blue:" << bgr.blue << ",bgr_green:" << bgr.green << ",bgr_red:" << bgr.red << endl;

//                cout << "bgr_blue:" << src_x << ",bgr_green:" << src_y << ",bgr_red:" << src_z << endl;
//                cout << "color1:" << color[0] << ",color2:" << color[1] << ",color3:" << color[2] << endl;
//                cout << "new intensity :" << intensity << endl;
//                cout << "flooding pixel :" << i << "," << j << endl;
//            }

        }
    }
    return src_img;
}

Mat src_gray;
int thresh = 100;
RNG rng(12345);

vector<vector<Point>> thresh_callback(Mat src_gray)
{
    Mat canny_output;
    Canny( src_gray, canny_output, thresh, thresh*2 );
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours( drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0 );
    }
//    for(int i = 0; i < contours.size(); i++){
//        cout << contours[i] << endl;
//    }
//    imshow( "Contours", drawing );
    return contours;
}

int opencv_findContours(String input_path, String output_path){
    Mat src = imread( input_path);
    Mat canny_output;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    float max_contourArea = 0;
    vector<Point> max_contour;
    int contour_idx = 0;


    if( src.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        return -1;
    }
    cvtColor( src, src_gray, COLOR_BGR2GRAY );
    blur( src_gray, src_gray, Size(3,3) );
    Canny( src_gray, canny_output, thresh, thresh*2 );
    findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );

    for( size_t i = 0; i< contours.size(); i++ )
    {
        if(contourArea(contours[i]) > max_contourArea){
            max_contourArea = contourArea(contours[i]);
            max_contour = contours[i];
            contour_idx = i;
        }
//        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
//        drawContours( drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0 );
    }
//    const char* source_window = "Source";
//    namedWindow( source_window );
//    imshow( source_window, src );
//    const int max_thresh = 255;
//    createTrackbar( "Canny thresh:", source_window, &thresh, max_thresh, thresh_callback );
//    waitKey();
//    for(int i = 0; i < contours.size(); i++){
//        cout << contours[i] << endl;
//    }
    drawContours(src, max_contour, contour_idx, (0, 0, 255), 10, LINE_8, hierarchy, 0);
    imwrite(output_path, src);
    cout << "Finished!" << endl;
    return 0;
}



int main() {
//    int pixel_int = 0;
//    typedef Point3_<uint8_t> Pixel;

    String img_path = "E:\\Users\\Kevin\\Clion_Projects\\Open_Test_Cpp\\test.jpg";
    opencv_findContours(img_path, "findcontours_result.jpg");
//    int flag = IMREAD_COLOR;
//    int flag = IMREAD_GRAYSCALE;
//    Mat img = readImg(img_path, flag);

//    Mat img = imread(img_path );
//    Mat bin;
//    cv::threshold(img, bin, 125, 255, THRESH_BINARY);
//    cout << img.channels() << endl;

// Loop Test
//    for(int i = 0; i < bin.rows; i++){
////        uchar* img_p = img.ptr<uchar>(i);
//
//        for(int j = 0; j < bin.cols; j++){
//            Scalar intensity=bin.at<uchar>(i,j);
//            if(!isnan(bin.at<float>(i,j))){
//                cout << bin.at<float>(i,j) << endl;
//            }
//            if( intensity[0] != 0){
//                cout << intensity[0] << endl;
//            }
//            cout << img_p[j] << endl;
//            if(img_p[j] != -1){
//                cout << img_p[j] << endl;
//            }
//        }
//    }
//    Mat canny_img = cannyImg(img);
//    Mat points_map = canny2points(canny_img);
//    Mat adjusted_img = points2adjusted(points_map);
//    Mat contours_img = drawContours(adjusted_img, img);
//    imwrite("points_map_test.jpg", points_map);
//    imwrite("canny_img_test.jpg", canny_img);
//    imwrite("adjusted_img_test.jpg", adjusted_img);
//    imwrite("contours_img_test.jpg", contours_img);

//    Mat points_map = canny2points_3channels(canny_img);
    // TODO drawContours change the color of white pixel into red


//    for(int r = 0; r < img_test.rows; ++r) {
//        for(int c = 0; c < img_test.cols; ++c) {
//            int blue = img_test.at<Vec3b>(r,c)[0];
//            int green = img_test.at<Vec3b>(r,c)[0];
//            int red = img_test.at<Vec3b>(r,c)[0];
//
//            if(blue == 0 && green == 0){
//                std::cout << "Pixel at position (x, y) : (" << c << ", " << r << ") =" <<
//                          img_test.at<Vec3b>(r,c) << std::endl;
//            }
//
//        }
//    }
//    Pixel* pixel = canny_img.ptr<Pixel>(0,0);

    // Naive pixel access
// Loop over all rows
//    for (int r = 0; r < canny_img.rows; r++)
//    {
//        // Loop over all columns
//        for ( int c = 0; c < canny_img.cols; c++)
//        {
//            // Obtain pixel at (r, c)
//            pixel_int = canny_img.at<uchar>(r, c);
//            if(pixel_int != 0 && pixel_int != 255){
//                cout << "pixel_value: " << pixel_int << endl;
//            }
//
//        }
//
//    }
// Parallel execution with function object.
//    struct Operator
//    {
//        void operator ()(Pixel &pixel, const int * position) const
//        {
//            pixel.
//            cout << "pixel_x: " << pixel.x <<  "pixel_y: " << pixel.y << "pixel_z: " << pixel.z << endl;
//            // Perform a simple threshold operation
//        }
//    };
//    canny_img.forEach<Pixel>(Operator());
//    Mat img = imread("E:\\Users\\Kevin\\Clion_Projects\\Open_Test_Cpp\\test.jpg", int flags=IMREAD_COLOR);
//    if (img.empty()) {
//        cout << "Error" << endl;
//        return -1;
//    }

}

