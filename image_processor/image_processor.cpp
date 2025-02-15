// USES WEBCAM AND LIVE IMAGES

// #include <iostream>
// #include <opencv2/opencv.hpp>
// #include <opencv2/core/cuda.hpp>
// #include <opencv2/cudaimgproc.hpp>
// #include <opencv2/cudaarithm.hpp>
// #include <opencv2/cudafilters.hpp>

// using namespace cv;
// using namespace std;

// int main() {
//     // Check if CUDA is available
//     if (cuda::getCudaEnabledDeviceCount() == 0) {
//         cerr << "Error: No CUDA-enabled device found!" << endl;
//         return -1;
//     }

//     // Open webcam
//     VideoCapture cap(0);
//     if (!cap.isOpened()) {
//         cerr << "Error: Could not open webcam!" << endl;
//         return -1;
//     }

//     Mat img;             // CPU Matrix image
//     cuda::GpuMat imgGpu; // GPU Matrix image

//     while (true) {
//         auto start = getTickCount(); // Start time measurement

//         cap.read(img); // Capture frame from webcam
//         if (img.empty()) {
//             cerr << "Error: Blank frame grabbed!" << endl;
//             break;
//         }

//         imgGpu.upload(img); // Send image to GPU

//         // Convert to grayscale
//         cuda::cvtColor(imgGpu, imgGpu, COLOR_BGR2GRAY);

//         // Apply Gaussian blur
//         auto gaussianFilter = cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(3, 3), 1);
//         gaussianFilter->apply(imgGpu, imgGpu);

//         imgGpu.download(img); // Get processed image back from GPU

//         // Calculate FPS
//         auto end = getTickCount();
//         double totalTime = (end - start) / getTickFrequency();
//         double fps = totalTime > 0 ? 1.0 / totalTime : 0.0; // Prevent division by zero

//         cout << "FPS: " << fps << endl;

//         // Display processed frame
//         imshow("CUDA Processed Frame", img);

//         // Exit on ESC key press
//         if (waitKey(1) == 27)
//             break;
//     }

//     cap.release();
//     destroyAllWindows();
//     return 0;
// }

// USES LOCAL IMAGES

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

using namespace cv;
using namespace std;

int main() {
    // Check if CUDA is available
    if (cuda::getCudaEnabledDeviceCount() == 0) {
        cerr << "Error: No CUDA-enabled device found!" << endl;
        return -1;
    }

    // Load image from file
    Mat img = imread("image.png"); // Change "input.jpg" to the actual file path
    if (img.empty()) {
        cerr << "Error: Could not load image!" << endl;
        return -1;
    }

    cuda::GpuMat imgGpu; // GPU Matrix image

    auto start = getTickCount(); // Start time measurement

    imgGpu.upload(img); // Send image to GPU

    // Convert to grayscale
    cuda::cvtColor(imgGpu, imgGpu, COLOR_BGR2GRAY);

    // Apply Gaussian blur
    auto gaussianFilter = cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(3, 3), 10);
    gaussianFilter->apply(imgGpu, imgGpu);

    imgGpu.download(img); // Get processed image back from GPU

    // Calculate FPS
    auto end = getTickCount();
    double totalTime = (end - start) / getTickFrequency();
    double fps = totalTime > 0 ? 1.0 / totalTime : 0.0; // Prevent division by zero

    cout << "Processing Time: " << totalTime << " sec, FPS: " << fps << endl;

    // Display processed image
    imshow("CUDA Processed Image", img);
    waitKey(0); // Wait for key press

    destroyAllWindows();
    return 0;
}
