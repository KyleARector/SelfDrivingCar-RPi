// OpenCVTest.cpp
// Basic test of OpenCV functionality in C++
// Streams video from first found camera to GUI
//
// Compiles with g++ $(pkg-config --libs --cflags opencv) -o OpenCVTest OpenCVTest.cpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

int main(int argc,char ** argv)
{
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "ERROR: Unable to open the camera" << endl;
        return 0;
    }

    // Set the capture to 60 FPS, 720p
    cap.set(CV_CAP_PROP_FPS, 60);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT , 720);

    Mat frame;
    cout << "Start grabbing, press a key on Live window to terminate" << endl;
    while(1) {
        cap >> frame;
        if (frame.empty()) {
            cerr << "ERROR: Unable to grab from the camera" << endl;
            break;
        }
        // Invert the captured frame
        // Used if the camera is mounted upside down from mirror
        // 1 = flip y; 0 = flip x; -1 = flip both
        flip(frame, frame, -1);

        // Allow window to be resized
        namedWindow("Stream", WINDOW_NORMAL);
        // Populate window with frame data
        imshow("Stream",frame);
        int key = cv::waitKey(5);
        key = (key==255) ? -1 : key;
        if (key>=0) {
            break;
        }
    }

    cout << "Closing the camera" << endl;
    cap.release();
    destroyAllWindows();
    return 0;
}
