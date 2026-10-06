#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <vector>
#include <onnxruntime_cxx_api.h>

cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));

cv::Mat resizeFrame(const cv::Mat &frame, int targetWidth, int targetHeight)
{
    cv::Mat resizedFrame;
    cv::resize(frame, resizedFrame, cv::Size(targetWidth, targetHeight));
    return resizedFrame;
}
cv::Mat convertToGrayscale(const cv::Mat &frame)
{
    cv::Mat grayFrame;
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    return grayFrame;
}

cv::Mat normalizeGrayscale(const cv::Mat &grayFrame)
{
    cv::Mat normalizedFrame;
    grayFrame.convertTo(normalizedFrame, CV_32F, 1.0 / 255.0);
    return normalizedFrame;
}
cv::Mat blurFrame(const cv::Mat &frame, int kernelSize)
{
    cv::Mat blurredFrame;
    cv::GaussianBlur(frame, blurredFrame, cv::Size(kernelSize, kernelSize), 0);
    return blurredFrame;
}
cv::Mat cannyEdgeDetection(const cv::Mat &frame, double threshold1, double threshold2)
{
    cv::Mat edges;
    cv::Canny(frame, edges, threshold1, threshold2);
    return edges;
}

cv::Mat enhanceWithCLAHE(
    const cv::Mat &resizedFrame,
    cv::Mat &labFrame,
    cv::Mat &originalL,
    cv::Mat &enhancedL)
{
    cv::cvtColor(resizedFrame, labFrame, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> labChannels;
    cv::split(labFrame, labChannels);
    originalL = labChannels[0];
    clahe->apply(originalL, enhancedL);

    labChannels[0] = enhancedL;
    cv::Mat enhancedLabFrame;
    cv::merge(labChannels, enhancedLabFrame);

    cv::Mat enhancedBGR;
    cv::cvtColor(enhancedLabFrame, enhancedBGR, cv::COLOR_Lab2BGR);
    return enhancedBGR;
}

cv::Mat preprocessFrame(
    const cv::Mat &frame,
    cv::Mat &resizedFrame,
    cv::Mat &labFrame,
    cv::Mat &originalL,
    cv::Mat &enhancedL)
{
    resizedFrame = resizeFrame(frame, 640, 360);
    return enhanceWithCLAHE(resizedFrame, labFrame, originalL, enhancedL);
}

double calculateTimeDifference(const std::chrono::steady_clock::time_point &start, const std::chrono::steady_clock::time_point &end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void printFrameInfo(const cv::Mat &frame, const std::string &frameName)
{
    std::cout << "\n--- " << frameName << " ---\n";
    std::cout << "Width: " << frame.cols << std::endl;
    std::cout << "Height: " << frame.rows << std::endl;
    std::cout << "Channels: " << frame.channels() << std::endl;
    std::cout << "Total pixels: " << frame.total() << std::endl;
    std::cout << "Type: " << frame.type() << std::endl;
}

void pixelValueAt(const cv::Mat &frame, int row, int col, const std::string &frameName)
{
    if (row < 0 || row >= frame.rows || col < 0 || col >= frame.cols)
    {
        std::cerr << "Error: Pixel coordinates out of bounds." << std::endl;
        return;
    }

    if (frame.channels() == 1)
    {
        unsigned char pixelValue = frame.at<unsigned char>(row, col);
        std::cout << "Pixel [" << row << "," << col << "] Value for " << frameName << ": " << static_cast<int>(pixelValue) << std::endl;
    }
    else if (frame.channels() == 3)
    {
        cv::Vec3b pixelValue = frame.at<cv::Vec3b>(row, col);
        std::cout << "Pixel [" << row << "," << col << "] Value for " << frameName << ": "
                  << "B=" << static_cast<int>(pixelValue[0]) << ", "
                  << "G=" << static_cast<int>(pixelValue[1]) << ", "
                  << "R=" << static_cast<int>(pixelValue[2]) << std::endl;
    }
    else
    {
        std::cerr << "Error: Unsupported number of channels." << std::endl;
    }
}
void printHistogram(const cv::Mat &hist, const std::string &histName)
{
    std::cout << "\n--- " << histName << " Histogram ---\n";
    for (int i = 0; i < hist.rows; ++i)
    {
        std::cout << "Bin " << i << ": " << hist.at<float>(i) << std::endl;
    }
}

void printHistogramComparison(const cv::Mat &originalHist, const cv::Mat &enhancedHist)
{
    std::cout << "\n--- Histogram Comparison ---\n";
    for (int i = 0; i < originalHist.rows; ++i)
    {
        std::cout << "Bin " << i << ": Original=" << originalHist.at<float>(i)
                  << ", Enhanced=" << enhancedHist.at<float>(i) << std::endl;
    }
}

void printCLAHEStats(const cv::Mat &originalL, const cv::Mat &enhancedL)
{
    cv::Scalar originalMean, originalStdDev;
    cv::meanStdDev(originalL, originalMean, originalStdDev);

    cv::Scalar enhancedMean, enhancedStdDev;
    cv::meanStdDev(enhancedL, enhancedMean, enhancedStdDev);

    std::cout << "\n--- L Channel Statistics ---\n";
    std::cout << "Original Mean: " << originalMean[0] << std::endl;
    std::cout << "Original StdDev: " << originalStdDev[0] << std::endl;
    std::cout << "Enhanced Mean: " << enhancedMean[0] << std::endl;
    std::cout << "Enhanced StdDev: " << enhancedStdDev[0] << std::endl;
}

cv::Mat prepareYOLOFrame(const cv::Mat &frame)
{
    cv::Mat resized;

    cv::resize(
        frame,
        resized,
        cv::Size(640, 640));

    cv::cvtColor(
        resized,
        resized,
        cv::COLOR_BGR2RGB);

    return resized;
}
cv::Mat convertToYOLOFloat(const cv::Mat &frame)
{
    cv::Mat floatFrame;

    frame.convertTo(
        floatFrame,
        CV_32FC3,
        1.0 / 255.0);

    return floatFrame;
}
std::vector<float> convertHWCtoCHW(const cv::Mat &frame)
{
    const int width = frame.cols;
    const int height = frame.rows;
    const int channels = frame.channels();

    std::vector<float> chw(
        channels * width * height);

    for (int c = 0; c < channels; ++c)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                chw[c * width * height +
                    y * width +
                    x] = frame.at<cv::Vec3f>(y, x)[c];
            }
        }
    }

    return chw;
}

int main()
{
    Ort::Env env(
        ORT_LOGGING_LEVEL_WARNING,
        "VisionPipeline");
    Ort::SessionOptions sessionOptions;

    Ort::Session session(
        env,
        "../yolo26n.onnx",
        sessionOptions);
    Ort::AllocatorWithDefaultOptions allocator;

    auto inputNameAllocated =
        session.GetInputNameAllocated(0, allocator);

    auto outputNameAllocated =
        session.GetOutputNameAllocated(0, allocator);

    const char *inputName =
        inputNameAllocated.get();

    const char *outputName =
        outputNameAllocated.get();

    std::cout << "YOLO Input Name: "
              << inputName << std::endl;

    std::cout << "YOLO Output Name: "
              << outputName << std::endl;
    auto inputShape =
        session.GetInputTypeInfo(0)
            .GetTensorTypeAndShapeInfo()
            .GetShape();

    std::cout << "YOLO Input Shape: ";

    for (auto dim : inputShape)
    {
        std::cout << dim << " ";
    }

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
    double averageCaptureTime = 0.0;
    double fps = 0.0;
    int captureSamples = 0;

    double totalGrayTime = 0.0;
    int graySamples = 0;

    double totalBlurTime = 0.0;
    int blurSamples = 0;

    double totalCannyTime = 0.0;
    int cannySamples = 0;

    double totalClaheTime = 0.0;
    int claheSamples = 0;
    double totalPreprocessTime = 0.0;
    int preprocessSamples = 0;

    while (true)
    {

        auto captureStart = std::chrono::steady_clock::now();

        cap >> frame;
        auto captureEnd = std::chrono::steady_clock::now();

        if (frame.empty())
        {
            std::cerr << "Error: Empty frame received." << std::endl;
            break;
        }
        cv::Mat yoloFrame = prepareYOLOFrame(frame);
        cv::Mat yoloFloatFrame =
            convertToYOLOFloat(yoloFrame);
        std::vector<float> inputTensorValues =
            convertHWCtoCHW(yoloFloatFrame);

        double captureTime =
            calculateTimeDifference(captureStart, captureEnd);
        totalCaptureTime += captureTime;
        captureSamples++;

        std::array<int64_t, 4> inputShape = {
            1, 3, 640, 640};

        Ort::MemoryInfo memoryInfo =
            Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault);

        Ort::Value inputTensor =
            Ort::Value::CreateTensor<float>(
                memoryInfo,
                inputTensorValues.data(),
                inputTensorValues.size(),
                inputShape.data(),
                inputShape.size());
        const char *inputNames[] = {
            inputName};

        const char *outputNames[] = {
            outputName};
        auto outputTensors = session.Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1);
        auto preprocessStart = std::chrono::steady_clock::now();
        cv::Mat resizedFrame;
        cv::Mat labFrame;
        cv::Mat originalL;
        cv::Mat enhancedL;
        cv::Mat enhancedBGR =
            preprocessFrame(frame, resizedFrame, labFrame, originalL, enhancedL);
        auto preprocessEnd = std::chrono::steady_clock::now();

        double preprocessTime =
            calculateTimeDifference(preprocessStart, preprocessEnd);
        totalPreprocessTime += preprocessTime;
        preprocessSamples++;

        auto grayStart = std::chrono::steady_clock::now();
        cv::Mat grayFrame = convertToGrayscale(resizedFrame);
        auto grayEnd = std::chrono::steady_clock::now();

        double grayTime =
            calculateTimeDifference(grayStart, grayEnd);
        totalGrayTime += grayTime;
        graySamples++;

        cv::Mat normalizedFrame = normalizeGrayscale(grayFrame);

        auto blurStart = std::chrono::steady_clock::now();
        cv::Mat blurredFrame = blurFrame(grayFrame, 5);
        auto blurEnd = std::chrono::steady_clock::now();

        double blurTime =
            calculateTimeDifference(blurStart, blurEnd);
        totalBlurTime += blurTime;
        blurSamples++;

        auto cannyStart = std::chrono::steady_clock::now();
        cv::Mat edges = cannyEdgeDetection(blurredFrame, 50, 150);

        auto cannyEnd = std::chrono::steady_clock::now();
        double cannyTime =
            calculateTimeDifference(cannyStart, cannyEnd);
        totalCannyTime += cannyTime;
        cannySamples++;

        auto claheStart = std::chrono::steady_clock::now();
        cv::Mat claheFrame;
        clahe->apply(grayFrame, claheFrame);
        auto claheEnd =
            std::chrono::steady_clock::now();

        double claheTime =
            calculateTimeDifference(claheStart, claheEnd);
        totalClaheTime += claheTime;
        claheSamples++;

        cv::Mat hsvFrame;
        cv::cvtColor(resizedFrame, hsvFrame, cv::COLOR_BGR2HSV);

        cv::Scalar originalMean;
        cv::Scalar originalStdDev;

        cv::Scalar enhancedMean;
        cv::Scalar enhancedStdDev;

        cv::meanStdDev(
            originalL,
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
            &originalL,
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

        frameCount++;

        // Inspect the first frame
        if (firstFrame)
        {
            printFrameInfo(frame, "Original Frame");
            printFrameInfo(resizedFrame, "Resized Frame");
            printFrameInfo(grayFrame, "Grayscale Frame");
            printFrameInfo(normalizedFrame, "Normalized Grayscale Frame");
            printFrameInfo(hsvFrame, "HSV Frame");
            printFrameInfo(labFrame, "LAB Frame");
            printFrameInfo(yoloFrame, "YOLO Input Frame");
            printFrameInfo(yoloFloatFrame, "YOLO Float Frame");
            printHistogram(originalHist, "Original");
            printHistogram(enhancedHist, "Enhanced");
            printHistogramComparison(originalHist, enhancedHist);
            printCLAHEStats(originalL, enhancedL);

            pixelValueAt(frame, 0, 0, "Original Frame");
            pixelValueAt(resizedFrame, 0, 0, "Resized Frame");
            pixelValueAt(grayFrame, 0, 0, "Grayscale Frame");
            pixelValueAt(normalizedFrame, 0, 0, "Normalized Grayscale Frame");
            pixelValueAt(hsvFrame, 0, 0, "HSV Frame");
            pixelValueAt(labFrame, 0, 0, "LAB Frame");
            std::cout << "YOLO Tensor Values: "
                      << inputTensorValues.size()
                      << std::endl;
            std::cout << "ONNX Tensor Created" << std::endl;
            auto outputInfo =
                outputTensors[0].GetTensorTypeAndShapeInfo();

            auto outputShape =
                outputInfo.GetShape();

            std::cout << "YOLO Output Shape: ";

            for (auto dim : outputShape)
            {
                std::cout << dim << " ";
            }
            float *outputData =
                outputTensors[0].GetTensorMutableData<float>();

            // std::cout << "First 10 output values:" << std::endl;

            // for (int i = 0; i < 10; ++i)
            // {
            //     std::cout << outputData[i] << " ";
            // }
            // std::cout << "\nFirst prediction:" << std::endl;

            // for (int i = 0; i < 84; ++i)
            // {
            //     std::cout << outputData[i * 8400] << " ";
            // }

            // std::cout << std::endl;
            float bestScore = 0.0f;
            int bestPrediction = -1;
            int bestClass = -1;

            for (int p = 0; p < 8400; ++p)
            {
                for (int c = 0; c < 80; ++c)
                {
                    float score = outputData[(4 + c) * 8400 + p];

                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestPrediction = p;
                        bestClass = c;
                    }
                }
            }

            std::cout << "Best Prediction: "
                      << bestPrediction << std::endl;

            std::cout << "Best Class: "
                      << bestClass << std::endl;

            std::cout << "Best Score: "
                      << bestScore << std::endl;
            float x1 = outputData[0 * 8400 + bestPrediction];
            float y1 = outputData[1 * 8400 + bestPrediction];
            float x2 = outputData[2 * 8400 + bestPrediction];
            float y2 = outputData[3 * 8400 + bestPrediction];

            std::cout << "Best Box:" << std::endl;

            std::cout << "x1: " << x1 << std::endl;
            std::cout << "y1: " << y1 << std::endl;
            std::cout << "x2: " << x2 << std::endl;
            std::cout << "y2: " << y2 << std::endl;

            std::cout << std::endl;
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
            fps = static_cast<double>(frameCount) / elapsedSeconds;
            averageCaptureTime = captureSamples > 0 ? totalCaptureTime / captureSamples : 0.0;

            double averageGrayTime =
                graySamples > 0 ? totalGrayTime / graySamples : 0.0;

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
                blurSamples > 0 ? totalBlurTime / blurSamples : 0.0;

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
                cannySamples > 0 ? totalCannyTime / cannySamples : 0.0;

            std::cout << " | Avg Canny: "
                      << averageCannyTime
                      << " ms"
                      << std::endl;

            double averageClaheTime =
                claheSamples > 0 ? totalClaheTime / claheSamples : 0.0;

            std::cout << " | Avg CLAHE: "
                      << averageClaheTime
                      << " ms"
                      << std::endl;

            double averagePreprocessTime =
                preprocessSamples > 0 ? totalPreprocessTime / preprocessSamples : 0.0;

            std::cout << " | Avg Preprocess: "
                      << averagePreprocessTime
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
            totalCaptureTime = 0.0;
            captureSamples = 0;
            totalPreprocessTime = 0.0;
            preprocessSamples = 0;
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