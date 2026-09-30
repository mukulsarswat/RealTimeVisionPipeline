#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

int main()
{

    std::cout << "OpenCV version: "
              << CV_VERSION << std::endl;

    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return 1;
    }

    std::cout << "Camera opened successfully!" << std::endl;
    std::cout << "Press 's' to save a frame." << std::endl;
    std::cout << "Press 'q' to quit." << std::endl;

    cv::Mat frame;

    // Used to print information only once
    bool firstFrame = true;
    int frameCount = 0;

    auto startTime = std::chrono::steady_clock::now();
    double totalCaptureTime = 0.0;
    int captureSamples = 0;

    double totalGrayTime = 0.0;
    int graySamples = 0;

    while (true)
    {

        auto captureStart = std::chrono::steady_clock::now();

        cap >> frame;
        cv::Mat resizedFrame;

        cv::resize(
            frame,
            resizedFrame,
            cv::Size(640, 360));

        cv::Mat grayFrame;

        auto grayStart = std::chrono::steady_clock::now();

        cv::cvtColor(
            resizedFrame,
            grayFrame,
            cv::COLOR_BGR2GRAY);

        auto grayEnd = std::chrono::steady_clock::now();

        double grayTime =
            std::chrono::duration<double, std::milli>(
                grayEnd - grayStart)
                .count();

        totalGrayTime += grayTime;
        graySamples++;

        auto captureEnd = std::chrono::steady_clock::now();

        double captureTime =
            std::chrono::duration<double, std::milli>(
                captureEnd - captureStart)
                .count();

        totalCaptureTime += captureTime;
        captureSamples++;

        if (frame.empty())
        {
            std::cerr << "Error: Empty frame received." << std::endl;
            break;
        }
        frameCount++;

        // Inspect the first frame
        if (firstFrame)
        {   std::cout << "\n--- Original Frame ---\n";

            std::cout << "Width: "
                      << frame.cols << std::endl;

            std::cout << "Height: "
                      << frame.rows << std::endl;
            
            std::cout << "Channels: "
                      << frame.channels() << std::endl;

            std::cout << "Total pixels: "
                      << frame.total() << std::endl;
            
            std::cout << "Type: "
                      << frame.type() << std::endl;


            std::cout << "\n--- Frame Information ---\n";

            std::cout << "Width: "
                      << resizedFrame.cols << std::endl;

            std::cout << "Height: "
                      << resizedFrame.rows << std::endl;

            std::cout << "Channels: "
                      << resizedFrame.channels() << std::endl;

            std::cout << "Total pixels: "
                      << resizedFrame.total() << std::endl;

            std::cout << "Type: "
                      << resizedFrame.type() << std::endl;


            std::cout << "\n--- Grayscale Frame ---\n";

            std::cout << "Width: "
                      << grayFrame.cols << std::endl;

            std::cout << "Height: "
                      << grayFrame.rows << std::endl;

            std::cout << "Channels: "
                      << grayFrame.channels() << std::endl;
            
            std::cout << "Total pixels: "
                      << grayFrame.total() << std::endl;

            std::cout << "Type: "
                      << grayFrame.type() << std::endl;
            
            cv::Vec3b pixelOriginal = frame.at<cv::Vec3b>(0, 0);
            std::cout << "Pixel [0,0](for original frame): "
                      << "B=" << static_cast<int>(pixelOriginal[0])
                      << " G=" << static_cast<int>(pixelOriginal[1])
                      << " R=" << static_cast<int>(pixelOriginal[2])
                      << std::endl;

            // Get pixel at row 0, column 0
            cv::Vec3b pixel = resizedFrame.at<cv::Vec3b>(0, 0);

            std::cout << "Pixel [0,0](for resized frame): "
                      << "B=" << static_cast<int>(pixel[0])
                      << " G=" << static_cast<int>(pixel[1])
                      << " R=" << static_cast<int>(pixel[2])
                      << std::endl;
            
            unsigned char pixelGray = grayFrame.at<unsigned char>(0, 0);

            std::cout << "Pixel [0,0](for grayscale frame): "
                      << "Gray=" << static_cast<int>(pixelGray)
                      << std::endl;

            firstFrame = false;
        }

        // Display live camera
        cv::imshow("Real-Time Vision Pipeline", grayFrame);

        char key = static_cast<char>(cv::waitKey(1));
        auto currentTime = std::chrono::steady_clock::now();

        double elapsedSeconds =
            std::chrono::duration<double>(
                currentTime - startTime)
                .count();

        if (elapsedSeconds >= 1.0)
        {

            double fps = frameCount / elapsedSeconds;

            double averageCaptureTime =
                totalCaptureTime / captureSamples;

            std::cout << "FPS: "
                      << fps
                      << " | Avg Capture: "
                      << averageCaptureTime
                      << " ms"
                      << std::endl;

            frameCount = 0;
            startTime = currentTime;

            totalCaptureTime = 0.0;
            captureSamples = 0;

            double averageGrayTime =
                totalGrayTime / graySamples;

            std::cout << "FPS: "
                      << fps
                      << " | Avg Capture: "
                      << averageCaptureTime
                      << " ms"
                      << " | Avg Grayscale: "
                      << averageGrayTime
                      << " ms"
                      << std::endl;

            totalGrayTime = 0.0;
            graySamples = 0;
        }

        // Save frame
        if (key == 's')
        {
            cv::imwrite("captured_frame_grayscale.jpg", grayFrame);
            std::cout << "Frame saved as captured_frame_grayscale.jpg"
                      << std::endl;
        }

        // Quit
        if (key == 'q')
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}