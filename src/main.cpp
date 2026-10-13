#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

#include <vector>
#include <onnxruntime_cxx_api.h>
#include <cstring>
#include <thread>
#include <mutex>
#include <atomic>
#include <coreml_provider_factory.h>

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

// cv::Mat prepareYOLOFrame(const cv::Mat &frame)
// {
//     cv::Mat resized;

//     cv::resize(
//         frame,
//         resized,
//         cv::Size(640, 640));

//     cv::cvtColor(
//         resized,
//         resized,
//         cv::COLOR_BGR2RGB);

//     return resized;
// }
// cv::Mat prepareYOLOFrame(const cv::Mat& frame)
// {
//     const int targetWidth = 640;
//     const int targetHeight = 640;

//     double scale = std::min(
//         static_cast<double>(targetWidth) / frame.cols,
//         static_cast<double>(targetHeight) / frame.rows
//     );

//     int newWidth =
//         static_cast<int>(frame.cols * scale);

//     int newHeight =
//         static_cast<int>(frame.rows * scale);

//     cv::Mat resized;

//     cv::resize(
//         frame,
//         resized,
//         cv::Size(newWidth, newHeight)
//     );

//     cv::Mat letterboxed(
//         targetHeight,
//         targetWidth,
//         resized.type(),
//         cv::Scalar(114, 114, 114)
//     );

//     int xOffset =
//         (targetWidth - newWidth) / 2;

//     int yOffset =
//         (targetHeight - newHeight) / 2;

//     resized.copyTo(
//         letterboxed(
//             cv::Rect(
//                 xOffset,
//                 yOffset,
//                 newWidth,
//                 newHeight
//             )
//         )
//     );

//     cv::cvtColor(
//         letterboxed,
//         letterboxed,
//         cv::COLOR_BGR2RGB
//     );

//     return letterboxed;
// }
struct LetterboxResult
{
    cv::Mat image;
    double scale;
    int xOffset;
    int yOffset;
};
LetterboxResult prepareYOLOFrame(const cv::Mat &frame)
{
    const int targetWidth = 640;
    const int targetHeight = 640;

    double scale = std::min(
        static_cast<double>(targetWidth) / frame.cols,
        static_cast<double>(targetHeight) / frame.rows);

    int newWidth =
        static_cast<int>(frame.cols * scale);

    int newHeight =
        static_cast<int>(frame.rows * scale);

    cv::Mat letterboxed(
        targetHeight,
        targetWidth,
        frame.type(),
        cv::Scalar(114, 114, 114));

    int xOffset = (targetWidth - newWidth) / 2;
    int yOffset = (targetHeight - newHeight) / 2;

    cv::Mat roi = letterboxed(
        cv::Rect(xOffset, yOffset, newWidth, newHeight));

    cv::resize(
        frame,
        roi,
        cv::Size(newWidth, newHeight));

    // cv::cvtColor(
    //     letterboxed,
    //     letterboxed,
    //     cv::COLOR_BGR2RGB);

    LetterboxResult result;

    result.image = letterboxed;
    result.scale = scale;
    result.xOffset = xOffset;
    result.yOffset = yOffset;

    return result;
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

void convertHWCtoCHW(
    const cv::Mat &frame,
    std::vector<cv::Mat> &channelMatrices,
    std::vector<float> &chw)
{
    const int width = frame.cols;
    const int height = frame.rows;
    const int channels = frame.channels();
    const int channelSize = width * height;

    // Reuse the channel matrix vector.
    cv::split(frame, channelMatrices);

    // Allocate only when the required size changes.
    const size_t requiredSize =
        static_cast<size_t>(channels) * channelSize;

    if (chw.size() != requiredSize)
    {
        chw.resize(requiredSize);
    }

    for (int c = 0; c < channels; ++c)
    {
        std::memcpy(
            chw.data() + c * channelSize,
            channelMatrices[channels - 1 - c].data,
            channelSize * sizeof(float));
    }
}

struct Detection
{
    float x;
    float y;
    float width;
    float height;
    float confidence;
    int classId;
};
Detection decodeBestDetection(
    const float *outputData,
    int predictionCount,
    int classCount)
{
    float bestScore = 0.0f;
    int bestPrediction = -1;
    int bestClass = -1;

    for (int p = 0; p < predictionCount; ++p)
    {
        for (int c = 0; c < classCount; ++c)
        {
            float score =
                outputData[(4 + c) * predictionCount + p];

            if (score > bestScore)
            {
                bestScore = score;
                bestPrediction = p;
                bestClass = c;
            }
        }
    }

    Detection detection;

    detection.x =
        outputData[0 * predictionCount + bestPrediction];

    detection.y =
        outputData[1 * predictionCount + bestPrediction];

    detection.width =
        outputData[2 * predictionCount + bestPrediction];

    detection.height =
        outputData[3 * predictionCount + bestPrediction];

    detection.confidence = bestScore;
    detection.classId = bestClass;

    return detection;
}
std::vector<Detection> decodeDetections(
    const float *outputData,
    int predictionCount,
    int classCount,
    float confidenceThreshold)
{
    std::vector<Detection> detections;

    for (int p = 0; p < predictionCount; ++p)
    {
        float bestScore = 0.0f;
        int bestClass = -1;

        for (int c = 0; c < classCount; ++c)
        {
            float score =
                outputData[(4 + c) * predictionCount + p];

            if (score > bestScore)
            {
                bestScore = score;
                bestClass = c;
            }
        }

        if (bestScore < confidenceThreshold)
        {
            continue;
        }

        Detection detection;

        detection.x =
            outputData[0 * predictionCount + p];

        detection.y =
            outputData[1 * predictionCount + p];

        detection.width =
            outputData[2 * predictionCount + p];

        detection.height =
            outputData[3 * predictionCount + p];

        detection.confidence = bestScore;
        detection.classId = bestClass;

        detections.push_back(detection);
    }

    return detections;
}
cv::Rect detectionToRect(const Detection &detection)
{
    int left =
        static_cast<int>(
            detection.x - detection.width / 2.0f);

    int top =
        static_cast<int>(
            detection.y - detection.height / 2.0f);

    int width =
        static_cast<int>(detection.width);

    int height =
        static_cast<int>(detection.height);

    return cv::Rect(left, top, width, height);
}
void drawDetection(
    cv::Mat &frame,
    const Detection &detection,
    const cv::Rect &box)
{
    cv::rectangle(
        frame,
        box,
        cv::Scalar(0, 255, 0),
        2);

    std::string label =
        "Class " +
        std::to_string(detection.classId) +
        " " +
        std::to_string(detection.confidence);

    cv::putText(
        frame,
        label,
        cv::Point(box.x, box.y - 10),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(0, 255, 0),
        2);
}
cv::Rect mapDetectionToOriginalFrame(
    const Detection &detection,
    const LetterboxResult &letterbox,
    const cv::Size &originalSize)
{
    float left =
        detection.x -
        detection.width / 2.0f;

    float top =
        detection.y -
        detection.height / 2.0f;

    float right =
        detection.x +
        detection.width / 2.0f;

    float bottom =
        detection.y +
        detection.height / 2.0f;

    left =
        (left - letterbox.xOffset) /
        letterbox.scale;

    top =
        (top - letterbox.yOffset) /
        letterbox.scale;

    right =
        (right - letterbox.xOffset) /
        letterbox.scale;

    bottom =
        (bottom - letterbox.yOffset) /
        letterbox.scale;

    left = std::max(0.0f, left);
    top = std::max(0.0f, top);

    right =
        std::min(
            static_cast<float>(originalSize.width),
            right);

    bottom =
        std::min(
            static_cast<float>(originalSize.height),
            bottom);

    return cv::Rect(
        static_cast<int>(left),
        static_cast<int>(top),
        static_cast<int>(right - left),
        static_cast<int>(bottom - top));
}

float calculateIoU(
    const Detection &a,
    const Detection &b)
{
    float aLeft =
        a.x - a.width / 2.0f;

    float aTop =
        a.y - a.height / 2.0f;

    float aRight =
        a.x + a.width / 2.0f;

    float aBottom =
        a.y + a.height / 2.0f;

    float bLeft =
        b.x - b.width / 2.0f;

    float bTop =
        b.y - b.height / 2.0f;

    float bRight =
        b.x + b.width / 2.0f;

    float bBottom =
        b.y + b.height / 2.0f;

    float intersectionLeft =
        std::max(aLeft, bLeft);

    float intersectionTop =
        std::max(aTop, bTop);

    float intersectionRight =
        std::min(aRight, bRight);

    float intersectionBottom =
        std::min(aBottom, bBottom);

    float intersectionWidth =
        std::max(
            0.0f,
            intersectionRight - intersectionLeft);

    float intersectionHeight =
        std::max(
            0.0f,
            intersectionBottom - intersectionTop);

    float intersectionArea =
        intersectionWidth * intersectionHeight;

    float areaA =
        a.width * a.height;

    float areaB =
        b.width * b.height;

    float unionArea =
        areaA + areaB - intersectionArea;

    if (unionArea <= 0.0f)
    {
        return 0.0f;
    }

    return intersectionArea / unionArea;
}
std::vector<Detection> applyNMS(
    const std::vector<Detection> &detections,
    float iouThreshold)
{
    std::vector<Detection> result;

    std::vector<bool> suppressed(
        detections.size(),
        false);

    for (size_t i = 0; i < detections.size(); ++i)
    {
        if (suppressed[i])
        {
            continue;
        }

        result.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j)
        {
            if (suppressed[j])
            {
                continue;
            }

            if (detections[i].classId !=
                detections[j].classId)
            {
                continue;
            }

            float iou =
                calculateIoU(
                    detections[i],
                    detections[j]);

            if (iou > iouThreshold)
            {
                suppressed[j] = true;
            }
        }
    }

    return result;
}
cv::Mat latestFrame;

std::mutex frameMutex;

std::atomic<bool> cameraRunning(true);
void cameraThread(cv::VideoCapture &cap)
{
    while (cameraRunning)
    {
        cv::Mat newFrame;

        if (!cap.read(newFrame))
        {
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(frameMutex);

            newFrame.copyTo(latestFrame);
        }
    }
}

int main()
{

    Ort::Env env(
        ORT_LOGGING_LEVEL_WARNING,
        "VisionPipeline");

    Ort::SessionOptions sessionOptions;

    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_CoreML(
            sessionOptions, COREML_FLAG_ENABLE_ON_SUBGRAPH));

    Ort::Session session(env, "../yolo26n.onnx", sessionOptions);

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
    cap.set(
        cv::CAP_PROP_FRAME_WIDTH,
        640);

    cap.set(
        cv::CAP_PROP_FRAME_HEIGHT,
        480);

    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return 1;
    }

    double cameraFPS =
        cap.get(cv::CAP_PROP_FPS);
    double cameraWidth =
        cap.get(cv::CAP_PROP_FRAME_WIDTH);

    double cameraHeight =
        cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::cout
        << "Camera Resolution: "
        << cameraWidth
        << " x "
        << cameraHeight
        << std::endl;
    std::cout
        << "Camera FPS: "
        << cameraFPS
        << std::endl;
    std::cout << "Camera opened successfully!" << std::endl;
    std::cout << "Press 's' to save a frame." << std::endl;
    std::cout << "Press 'q' to quit." << std::endl;

    // cv::Mat frame;

    // Used to print information only once
    // bool firstFrame = true;
    // int frameCount = 0;

    // auto startTime = std::chrono::steady_clock::now();
    double totalCaptureTime = 0.0;
    int captureSamples = 0;

    // double totalGrayTime = 0.0;
    // int graySamples = 0;

    // double totalBlurTime = 0.0;
    // int blurSamples = 0;

    // double totalCannyTime = 0.0;
    // int cannySamples = 0;

    // double totalClaheTime = 0.0;
    // int claheSamples = 0;
    // double totalPreprocessTime = 0.0;
    // int preprocessSamples = 0;

    int processedFrame = 0;
    int inferenceSamples = 0;
    double totalInferenceTime = 0.0;

    std::thread camera(
        cameraThread,
        std::ref(cap));

    auto fpsStart = std::chrono::steady_clock::now();
    while (true)
    {
        auto frameStart = std::chrono::steady_clock::now();

        auto captureStart = std::chrono::steady_clock::now();

        // cap.read(frame);
        cv::Mat frame;

        {
            std::lock_guard<std::mutex> lock(frameMutex);

            if (!latestFrame.empty())
            {
                latestFrame.copyTo(frame);
            }
        }

        if (frame.empty())
        {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(1));
            continue;
        }
        auto captureEnd = std::chrono::steady_clock::now();

        double captureTime =
            calculateTimeDifference(captureStart, captureEnd);

        std::cout << "Capture Time: "
                  << captureTime
                  << " ms"
                  << std::endl;

        if (frame.empty())
        {
            std::cerr << "Error: Empty frame received." << std::endl;
            break;
        }
        // cv::Mat yoloFrame = prepareYOLOFrame(frame);
        auto yoloPreprocessStart =
            std::chrono::steady_clock::now();

        auto letterboxStart =
            std::chrono::steady_clock::now();

        LetterboxResult letterbox =
            prepareYOLOFrame(frame);

        auto letterboxEnd =
            std::chrono::steady_clock::now();
        double letterboxTime =
            std::chrono::duration<double, std::milli>(
                letterboxEnd - letterboxStart)
                .count();

        cv::Mat yoloFrame =
            letterbox.image;

        auto floatStart =
            std::chrono::steady_clock::now();

        cv::Mat yoloFloatFrame =
            convertToYOLOFloat(yoloFrame);

        auto floatEnd =
            std::chrono::steady_clock::now();
        double floatTime =
            std::chrono::duration<double, std::milli>(
                floatEnd - floatStart)
                .count();

        auto chwStart =
            std::chrono::steady_clock::now();

        static std::vector<cv::Mat> channelMatrices;
        static std::vector<float> inputTensorValues;

        convertHWCtoCHW(yoloFloatFrame, channelMatrices, inputTensorValues);

        auto chwEnd =
            std::chrono::steady_clock::now();

        double chwTime =
            std::chrono::duration<double, std::milli>(
                chwEnd - chwStart)
                .count();

        auto yoloPreprocessEnd =
            std::chrono::steady_clock::now();

        double yoloPreprocessTime =
            std::chrono::duration<double, std::milli>(
                yoloPreprocessEnd - yoloPreprocessStart)
                .count();

        std::cout << "YOLO Preprocess Time: "
                  << yoloPreprocessTime
                  << " ms"
                  << std::endl;
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

        auto inferenceStart =
            std::chrono::high_resolution_clock::now();

        auto outputTensors = session.Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1);

        auto inferenceEnd =
            std::chrono::high_resolution_clock::now();

        double inferenceTime =
            std::chrono::duration<double, std::milli>(
                inferenceEnd - inferenceStart)
                .count();
        double currentInferenceTime =
            std::chrono::duration<double, std::milli>(
                inferenceEnd - inferenceStart)
                .count();

        std::cout
            << "Current Inference: "
            << currentInferenceTime
            << " ms"
            << std::endl;

        auto outputInfo =
            outputTensors[0].GetTensorTypeAndShapeInfo();

        auto outputShape =
            outputInfo.GetShape();

        float *outputData =
            outputTensors[0].GetTensorMutableData<float>();

        auto decodeStart =
            std::chrono::high_resolution_clock::now();

        std::vector<Detection> detections =
            decodeDetections(
                outputData,
                8400,
                80,
                0.50f);

        auto decodeEnd =
            std::chrono::high_resolution_clock::now();

        double decodeTime =
            std::chrono::duration<double, std::milli>(
                decodeEnd - decodeStart)
                .count();
        std::cout << "Decode Time: "
                  << decodeTime
                  << " ms"
                  << std::endl;
        auto nmsStart =
            std::chrono::high_resolution_clock::now();

        std::vector<Detection> finalDetections =
            applyNMS(
                detections,
                0.45f);

        auto nmsEnd =
            std::chrono::high_resolution_clock::now();

        double nmsTime =
            std::chrono::duration<double, std::milli>(
                nmsEnd - nmsStart)
                .count();

        std::cout
            << "NMS: "
            << nmsTime
            << " ms"
            << std::endl;
        auto drawStart =
            std::chrono::high_resolution_clock::now();

        for (const Detection &detection : finalDetections)
        {
            cv::Rect box =
                mapDetectionToOriginalFrame(
                    detection,
                    letterbox,
                    frame.size());

            drawDetection(
                frame,
                detection,
                box);
        }

        auto drawEnd =
            std::chrono::high_resolution_clock::now();

        double drawTime =
            std::chrono::duration<double, std::milli>(
                drawEnd - drawStart)
                .count();

        std::cout
            << "Drawing: "
            << drawTime
            << " ms"
            << std::endl;
        totalInferenceTime += inferenceTime;
        processedFrame++;
        inferenceSamples++;

        // std::cout << "Inference: "
        //           << inferenceTime
        //           << " ms"
        //           << std::endl;
        // auto preprocessStart = std::chrono::steady_clock::now();
        // cv::Mat resizedFrame;
        // cv::Mat labFrame;
        // cv::Mat originalL;
        // cv::Mat enhancedL;
        // cv::Mat enhancedBGR =
        //     preprocessFrame(frame, resizedFrame, labFrame, originalL, enhancedL);
        // auto preprocessEnd = std::chrono::steady_clock::now();

        // double preprocessTime =
        //     calculateTimeDifference(preprocessStart, preprocessEnd);
        // totalPreprocessTime += preprocessTime;
        // preprocessSamples++;

        // auto grayStart = std::chrono::steady_clock::now();
        // cv::Mat grayFrame = convertToGrayscale(resizedFrame);
        // auto grayEnd = std::chrono::steady_clock::now();

        // double grayTime =
        //     calculateTimeDifference(grayStart, grayEnd);
        // totalGrayTime += grayTime;
        // graySamples++;

        // cv::Mat normalizedFrame = normalizeGrayscale(grayFrame);

        // auto blurStart = std::chrono::steady_clock::now();
        // cv::Mat blurredFrame = blurFrame(grayFrame, 5);
        // auto blurEnd = std::chrono::steady_clock::now();

        // double blurTime =
        //     calculateTimeDifference(blurStart, blurEnd);
        // totalBlurTime += blurTime;
        // blurSamples++;

        // auto cannyStart = std::chrono::steady_clock::now();
        // cv::Mat edges = cannyEdgeDetection(blurredFrame, 50, 150);

        // auto cannyEnd = std::chrono::steady_clock::now();
        // double cannyTime =
        //     calculateTimeDifference(cannyStart, cannyEnd);
        // totalCannyTime += cannyTime;
        // cannySamples++;

        // auto claheStart = std::chrono::steady_clock::now();
        // cv::Mat claheFrame;
        // clahe->apply(grayFrame, claheFrame);
        // auto claheEnd =
        //     std::chrono::steady_clock::now();

        // double claheTime =
        //     calculateTimeDifference(claheStart, claheEnd);
        // totalClaheTime += claheTime;
        // claheSamples++;

        // cv::Mat hsvFrame;
        // cv::cvtColor(resizedFrame, hsvFrame, cv::COLOR_BGR2HSV);

        // cv::Scalar originalMean;
        // cv::Scalar originalStdDev;

        // cv::Scalar enhancedMean;
        // cv::Scalar enhancedStdDev;

        // cv::meanStdDev(
        //     originalL,
        //     originalMean,
        //     originalStdDev);

        // cv::meanStdDev(
        //     enhancedL,
        //     enhancedMean,
        //     enhancedStdDev);

        // cv::Mat originalHist;
        // cv::Mat enhancedHist;
        // int histSize = 256;

        // float range[] = {0, 256};
        // const float *histRange = {range};

        // cv::calcHist(
        //     &originalL,
        //     1,
        //     nullptr,
        //     cv::Mat(),
        //     originalHist,
        //     1,
        //     &histSize,
        //     &histRange);

        // cv::calcHist(
        //     &enhancedL,
        //     1,
        //     nullptr,
        //     cv::Mat(),
        //     enhancedHist,
        //     1,
        //     &histSize,
        //     &histRange);
        // int ranges[][2] = {
        //     {0, 49},
        //     {50, 99},
        //     {100, 149},
        //     {150, 199},
        //     {200, 255}};

        // frameCount++;

        // Inspect the first frame
        // if (firstFrame)
        // {
        //     // printFrameInfo(frame, "Original Frame");
        //     // printFrameInfo(resizedFrame, "Resized Frame");
        //     // printFrameInfo(grayFrame, "Grayscale Frame");
        //     // printFrameInfo(normalizedFrame, "Normalized Grayscale Frame");
        //     // printFrameInfo(hsvFrame, "HSV Frame");
        //     // printFrameInfo(labFrame, "LAB Frame");
        //     // printFrameInfo(yoloFrame, "YOLO Input Frame");
        //     // printFrameInfo(yoloFloatFrame, "YOLO Float Frame");
        //     // printHistogram(originalHist, "Original");
        //     // printHistogram(enhancedHist, "Enhanced");
        //     // printHistogramComparison(originalHist, enhancedHist);
        //     // printCLAHEStats(originalL, enhancedL);

        //     // pixelValueAt(frame, 0, 0, "Original Frame");
        //     // pixelValueAt(resizedFrame, 0, 0, "Resized Frame");
        //     // pixelValueAt(grayFrame, 0, 0, "Grayscale Frame");
        //     // pixelValueAt(normalizedFrame, 0, 0, "Normalized Grayscale Frame");
        //     // pixelValueAt(hsvFrame, 0, 0, "HSV Frame");
        //     // pixelValueAt(labFrame, 0, 0, "LAB Frame");
        //     // std::cout << "YOLO Tensor Values: "
        //     //           << inputTensorValues.size()
        //     //           << std::endl;
        //     // std::cout << "ONNX Tensor Created" << std::endl;

        //     // std::cout << "Detections above threshold: "
        //     //           << detections.size()
        //     //           << std::endl;

        //     // for (const Detection &detection : detections)
        //     // {
        //     //     std::cout << "Class: "
        //     //               << detection.classId
        //     //               << " | Confidence: "
        //     //               << detection.confidence
        //     //               << " | X: "
        //     //               << detection.x
        //     //               << " | Y: "
        //     //               << detection.y
        //     //               << " | W: "
        //     //               << detection.width
        //     //               << " | H: "
        //     //               << detection.height
        //     //               << std::endl;
        //     // }

        //     // std::cout << "Raw detections: "
        //     //           << detections.size()
        //     //           << std::endl;

        //     // std::cout << "After NMS: "
        //     //           << finalDetections.size()
        //     //           << std::endl;

        //     // std::cout << "First 10 output values:" << std::endl;

        //     // for (int i = 0; i < 10; ++i)
        //     // {
        //     //     std::cout << outputData[i] << " ";
        //     // }
        //     // std::cout << "\nFirst prediction:" << std::endl;

        //     // for (int i = 0; i < 84; ++i)
        //     // {
        //     //     std::cout << outputData[i * 8400] << " ";
        //     // }

        //     // std::cout << std::endl;
        //     // Detection bestDetection =
        //     //     decodeBestDetection(
        //     //         outputData,
        //     //         8400,
        //     //         80);
        //     // cv::Rect originalBox =
        //     //     mapDetectionToOriginalFrame(
        //     //         bestDetection,
        //     //         letterbox,
        //     //         frame.size());

        //     // std::cout << "Best Detection:" << std::endl;

        //     // std::cout << "X: "
        //     //           << bestDetection.x << std::endl;

        //     // std::cout << "Y: "
        //     //           << bestDetection.y << std::endl;

        //     // std::cout << "Width: "
        //     //           << bestDetection.width << std::endl;

        //     // std::cout << "Height: "
        //     //           << bestDetection.height << std::endl;

        //     // std::cout << "Confidence: "
        //     //           << bestDetection.confidence << std::endl;

        //     // std::cout << "Class ID: "
        //     //           << bestDetection.classId << std::endl;

        //     // cv::Rect bestBox =
        //     //     detectionToRect(bestDetection);

        //     // std::cout << "Box:" << std::endl;

        //     // std::cout << "Left: "
        //     //           << bestBox.x << std::endl;

        //     // std::cout << "Top: "
        //     //           << bestBox.y << std::endl;

        //     // std::cout << "Width: "
        //     //           << bestBox.width << std::endl;

        //     // std::cout << "Height: "
        //     //           << bestBox.height << std::endl;
        //     // drawDetection(
        //     //     frame,
        //     //     bestDetection,
        //     //     originalBox);

        //     // cv::imshow(
        //     //     "YOLO Detection",
        //     //     frame);

        //     std::cout << std::endl;
        //     firstFrame = false;
        // }

        // Display live camera
        // cv::rectangle(
        //     frame,
        //     cv::Rect(100, 100, 300, 300),
        //     cv::Scalar(0, 255, 0),
        //     3
        // );

        auto imshowStart = std::chrono::steady_clock::now();

        cv::imshow("YOLO Detection", frame);

        auto imshowEnd = std::chrono::steady_clock::now();

        int key = cv::waitKey(1);

        auto waitKeyEnd = std::chrono::steady_clock::now();

        double imshowMs =
            std::chrono::duration<double, std::milli>(
                imshowEnd - imshowStart)
                .count();

        double waitKeyMs =
            std::chrono::duration<double, std::milli>(
                waitKeyEnd - imshowEnd)
                .count();

        std::cout << "imshow: " << imshowMs
                  << " ms | waitKey: " << waitKeyMs << " ms\n";

        // processedFrames++;

        // if (elapsed >= 0.5)
        // {
        //     double fps = processedFrames / elapsed;

        //     std::cout << "-------------------------------------------------------------Processing FPS: " << fps << '\n';

        //     processedFrames = 0;
        //     fpsStart = now1;
        // }
        auto now = std::chrono::steady_clock::now();

        double elapseSeconds =
            std::chrono::duration<double>(
                now - fpsStart)
                .count();

        double fps = 30.0 / elapseSeconds;

        double averageInference =
            totalInferenceTime / 30.0;

        if (elapseSeconds >= 0.5)
        {
            double fps = processedFrame / elapseSeconds;

            double averageInference =
                inferenceSamples > 0
                    ? totalInferenceTime / inferenceSamples
                    : 0.0;

            std::cout << "Processing FPS: " << fps
                      << " | Avg Inference: "
                      << averageInference << " ms\n";

            processedFrame = 0;
            inferenceSamples = 0;
            totalInferenceTime = 0.0;
            fpsStart = now;
        }

        // double displayTime =
        //     std::chrono::duration<double, std::milli>(
        //         displayEnd - displayStart)
        //         .count();

        // std::cout
        //     << "Display: "
        //     << displayTime
        //     << " ms"
        // << std::endl;
        // cv::imshow("Real-Time Vision Pipeline", frame);

        // char key = static_cast<char>(cv::waitKey(1));
        auto frameEnd = std::chrono::steady_clock::now();

        double frameTime =
            std::chrono::duration<double, std::milli>(
                frameEnd - frameStart)
                .count();
        auto currentTime = std::chrono::steady_clock::now();

        double elapsedSeconds =
            std::chrono::duration<double>(
                currentTime - fpsStart)
                .count();
        // if (frameCount1 % 30 == 0)
        // {
        //     std::cout << "Total Frame Time: "
        //               << frameTime
        //               << " ms"
        //               << std::endl;
        // }

        // if (elapsedSeconds >= 1.0)
        // {
        //     fps = / elapsedSeconds;
        //     averageCaptureTime = captureSamples > 0 ? totalCaptureTime / captureSamples : 0.0;

        //     // double averageGrayTime =
        //     //     graySamples > 0 ? totalGrayTime / graySamples : 0.0;

        //     // std::cout << "FPS: "
        //     //           << fps
        //     //           << " | Avg Capture: "
        //     //           << averageCaptureTime
        //     //           << " ms"
        //     //           << " | Avg Grayscale: "
        //     //           << averageGrayTime
        //     //           << " ms"
        //     //           << std::endl;

        //     // double averageBlurTime =
        //     //     blurSamples > 0 ? totalBlurTime / blurSamples : 0.0;

        //     // std::cout << "FPS: "
        //     //           << fps
        //     //           << " | Avg Capture: "
        //     //           << averageCaptureTime
        //     //           << " ms"
        //     //           << " | Avg Grayscale: "
        //     //           << averageGrayTime
        //     //           << " ms"
        //     //           << " | Avg Blur: "
        //     //           << averageBlurTime
        //     //           << " ms"
        //     //           << std::endl;

        //     // double averageCannyTime =
        //     //     cannySamples > 0 ? totalCannyTime / cannySamples : 0.0;

        //     // std::cout << " | Avg Canny: "
        //     //           << averageCannyTime
        //     //           << " ms"
        //     //           << std::endl;

        //     // double averageClaheTime =
        //     //     claheSamples > 0 ? totalClaheTime / claheSamples : 0.0;

        //     // std::cout << " | Avg CLAHE: "
        //     //           << averageClaheTime
        //     //           << " ms"
        //     //           << std::endl;

        //     // std::cout << " | Avg Preprocess: "
        //     //           << averagePreprocessTime
        //     //           << " ms"
        //     //           << std::endl;

        //     // totalGrayTime = 0.0;
        //     // graySamples = 0;
        //     // totalBlurTime = 0.0;
        //     // blurSamples = 0;
        //     // totalCannyTime = 0.0;
        //     // cannySamples = 0;
        //     // totalClaheTime = 0.0;
        //     // claheSamples = 0;
        //     totalCaptureTime = 0.0;
        //     captureSamples = 0;
        // }

        // Save frame
        if (key == 's')
        {
            cv::imwrite("original_frame_BGR.jpg", frame);
            std::cout << "Frame saved as original_frame_BGR.jpg"
                      << std::endl;
        }

        // Quit
        if (key == 'q')
        {
            break;
        }
    }
    cameraRunning = false;

    if (camera.joinable())
    {
        camera.join();
    }

    cap.release();

    cv::destroyAllWindows();

    return 0;
}