#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <vector>

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

    double totalBlurTime = 0.0;
    int blurSamples = 0;

    double totalCannyTime = 0.0;
    int cannySamples = 0;

    double totalClaheTime = 0.0;
    int claheSamples = 0;

    while (true)
    {

        auto captureStart = std::chrono::steady_clock::now();

        cap >> frame;
        auto captureEnd = std::chrono::steady_clock::now();

        double captureTime =
            std::chrono::duration<double, std::milli>(
                captureEnd - captureStart)
                .count();

        totalCaptureTime += captureTime;
        captureSamples++;

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

        cv::Mat normalizedFrame;

        grayFrame.convertTo(
            normalizedFrame,
            CV_32F,
            1.0 / 255.0);

        auto grayEnd = std::chrono::steady_clock::now();

        double grayTime =
            std::chrono::duration<double, std::milli>(
                grayEnd - grayStart)
                .count();

        totalGrayTime += grayTime;
        graySamples++;

        auto blurStart = std::chrono::steady_clock::now();
        cv::Mat blurredFrame;
        cv::GaussianBlur(
            grayFrame,
            blurredFrame,
            cv::Size(5, 5),
            0);

        auto blurEnd = std::chrono::steady_clock::now();

        double blurTime =
            std::chrono::duration<double, std::milli>(
                blurEnd - blurStart)
                .count();
        totalBlurTime += blurTime;
        blurSamples++;

        cv::Mat edges;

        auto cannyStart = std::chrono::steady_clock::now();
        cv::Canny(
            blurredFrame,
            edges,
            50,
            150);

        auto cannyEnd = std::chrono::steady_clock::now();
        double cannyTime =
            std::chrono::duration<double, std::milli>(
                cannyEnd - cannyStart)
                .count();
        totalCannyTime += cannyTime;
        cannySamples++;

        cv::Mat enhancedFrame;
        cv::Ptr<cv::CLAHE> clahe =
            cv::createCLAHE();

        clahe->setClipLimit(4.0);
        clahe->setTilesGridSize(cv::Size(8, 8));
        auto claheStart = std::chrono::steady_clock::now();

        clahe->apply(
            grayFrame,
            enhancedFrame);
        auto claheEnd =
            std::chrono::steady_clock::now();

        double claheTime =
            std::chrono::duration<double, std::milli>(
                claheEnd - claheStart)
                .count();
        totalClaheTime += claheTime;
        claheSamples++;

        cv::Mat hsvFrame;

        cv::cvtColor(
            resizedFrame,
            hsvFrame,
            cv::COLOR_BGR2HSV);

        cv::Mat labFrame;
        cv::cvtColor(
            resizedFrame,
            labFrame,
            cv::COLOR_BGR2Lab);

        std::vector<cv::Mat> labChannels;
        cv::split(
            labFrame,
            labChannels);
        cv::Mat enhancedL;
        clahe->apply(
            labChannels[0],
            enhancedL);

        std::vector<cv::Mat> enhancedLabChannels = {
            enhancedL,
            labChannels[1],
            labChannels[2]};

        cv::Mat enhancedLabFrame;
        cv::merge(
            enhancedLabChannels,
            enhancedLabFrame);
        cv::Mat enhancedBGR;
        cv::cvtColor(
            enhancedLabFrame,
            enhancedBGR,
            cv::COLOR_Lab2BGR);
        cv::Scalar originalMean;
        cv::Scalar originalStdDev;

        cv::Scalar enhancedMean;
        cv::Scalar enhancedStdDev;

        cv::meanStdDev(
            labChannels[0],
            originalMean,
            originalStdDev);

        cv::meanStdDev(
            enhancedL,
            enhancedMean,
            enhancedStdDev);

        cv::Mat originalHist;
        cv::Mat enhancedHist;
        int histSize = 256;

        float range[] = {0, 256};
        const float *histRange = {range};

        cv::calcHist(
            &labChannels[0],
            1,
            nullptr,
            cv::Mat(),
            originalHist,
            1,
            &histSize,
            &histRange);

        cv::calcHist(
            &enhancedL,
            1,
            nullptr,
            cv::Mat(),
            enhancedHist,
            1,
            &histSize,
            &histRange);
        int ranges[][2] = {
            {0, 49},
            {50, 99},
            {100, 149},
            {150, 199},
            {200, 255}};

        if (frame.empty())
        {
            std::cerr << "Error: Empty frame received." << std::endl;
            break;
        }
        frameCount++;

        // Inspect the first frame
        if (firstFrame)
        {
            std::cout << "\n--- Original Frame ---\n";

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

            std::cout << "\n--- Normalized Grayscale Frame ---\n";
            std::cout << "Width: "
                      << normalizedFrame.cols << std::endl;
            std::cout << "Height: "
                      << normalizedFrame.rows << std::endl;
            std::cout << "Channels: "
                      << normalizedFrame.channels() << std::endl;
            std::cout << "Total pixels: "
                      << normalizedFrame.total() << std::endl;
            std::cout << "Type: "
                      << normalizedFrame.type() << std::endl
                      << std::endl;

            std::cout << "\n--- HSV Frame ---\n";

            std::cout << "Width: "
                      << hsvFrame.cols << std::endl;

            std::cout << "Height: "
                      << hsvFrame.rows << std::endl;

            std::cout << "Channels: "
                      << hsvFrame.channels() << std::endl;

            std::cout << "Type: "
                      << hsvFrame.type() << std::endl;

            std::cout << "\n--- LAB Frame ---\n";

            std::cout << "Width: "
                      << labFrame.cols << std::endl;

            std::cout << "Height: "
                      << labFrame.rows << std::endl;

            std::cout << "Channels: "
                      << labFrame.channels() << std::endl;

            std::cout << "Type: "
                      << labFrame.type() << std::endl;
            std::cout << "\n--- Histogram Comparison ---\n";

            for (int i = 0; i < 5; i++)
            {

                int start = ranges[i][0];
                int end = ranges[i][1];

                float originalCount = 0.0f;
                float enhancedCount = 0.0f;

                for (int j = start; j <= end; j++)
                {
                    originalCount += originalHist.at<float>(j);
                    enhancedCount += enhancedHist.at<float>(j);
                }

                std::cout << start << "-" << end
                          << " | Original: " << originalCount
                          << " | Enhanced: " << enhancedCount
                          << std::endl;
            }

            std::cout << "\n--- L Channel Statistics ---\n";

            std::cout << "Original Mean: "
                      << originalMean[0] << std::endl;

            std::cout << "Original StdDev: "
                      << originalStdDev[0] << std::endl;

            std::cout << "Enhanced Mean: "
                      << enhancedMean[0] << std::endl;

            std::cout << "Enhanced StdDev: "
                      << enhancedStdDev[0] << std::endl;

            cv::Vec3b pixelOriginal = frame.at<cv::Vec3b>(0, 0);
            std::cout << "\n Pixel [0,0](for original frame): "
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

            std::cout << "Pixel [0,0](for normalized grayscale frame): "
                      << "Gray=" << normalizedFrame.at<float>(0, 0)
                      << std::endl;

            cv::Vec3b hsvPixel =
                hsvFrame.at<cv::Vec3b>(0, 0);

            std::cout << "Pixel [0,0](for HSV frame): "
                      << "H=" << static_cast<int>(hsvPixel[0])
                      << " S=" << static_cast<int>(hsvPixel[1])
                      << " V=" << static_cast<int>(hsvPixel[2])
                      << std::endl;

            std::cout << "Pixel [0,0](for LAB frame): "
                      << "L=" << static_cast<int>(labFrame.at<cv::Vec3b>(0, 0)[0])
                      << " A=" << static_cast<int>(labFrame.at<cv::Vec3b>(0, 0)[1])
                      << " B=" << static_cast<int>(labFrame.at<cv::Vec3b>(0, 0)[2])
                      << std::endl;

            firstFrame = false;
        }

        // Display live camera
        cv::imshow("Real-Time Vision Pipeline", enhancedBGR);

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

            double averageBlurTime =
                totalBlurTime / blurSamples;

            std::cout << "FPS: "
                      << fps
                      << " | Avg Capture: "
                      << averageCaptureTime
                      << " ms"
                      << " | Avg Grayscale: "
                      << averageGrayTime
                      << " ms"
                      << " | Avg Blur: "
                      << averageBlurTime
                      << " ms"
                      << std::endl;

            double averageCannyTime =
                totalCannyTime / cannySamples;

            std::cout << " | Avg Canny: "
                      << averageCannyTime
                      << " ms"
                      << std::endl;

            double averageClaheTime =
                totalClaheTime / claheSamples;

            std::cout << " | Avg CLAHE: "
                      << averageClaheTime
                      << " ms"
                      << std::endl;

            totalGrayTime = 0.0;
            graySamples = 0;
            totalBlurTime = 0.0;
            blurSamples = 0;
            totalCannyTime = 0.0;
            cannySamples = 0;
            totalClaheTime = 0.0;
            claheSamples = 0;
        }

        // Save frame
        if (key == 's')
        {
            cv::imwrite("original_frame_BGR.jpg", resizedFrame);
            std::cout << "Frame saved as original_frame_BGR.jpg"
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