# Real-Time Vision Pipeline

A small C++17 and OpenCV application that reads frames from the default camera, inspects their image properties, resizes them, converts them to grayscale, displays the live result, and reports timing and frame-rate measurements.

## Pipeline

The application is implemented in [`src/main.cpp`](src/main.cpp) and runs the following stages for each camera frame:

1. **Camera acquisition**: `cv::VideoCapture cap(0)` opens camera device 0. `cap >> frame` retrieves a frame. The program exits with an error if the camera cannot be opened, and leaves the loop if an empty frame is received.
2. **Original-frame inspection**: on the first valid frame, the program prints its width, height, channel count, pixel count, OpenCV matrix type, and the BGR values at pixel `[0,0]`.
3. **Resize**: `cv::resize(frame, resizedFrame, cv::Size(640, 360))` creates a 640×360 color image. This is a fixed output size rather than an aspect-ratio-preserving fit.
4. **Grayscale conversion**: `cv::cvtColor(resizedFrame, grayFrame, cv::COLOR_BGR2GRAY)` converts the resized 3-channel BGR image to a one-channel grayscale image. The first-frame diagnostic also reports the grayscale image properties and pixel `[0,0]`.
5. **Display and keyboard input**: `cv::imshow` displays the grayscale frame. `cv::waitKey(1)` processes window events and checks for `s` or `q`.
6. **Optional save**: pressing `s` calls `cv::imwrite` to write the current grayscale frame to `captured_frame_grayscale.jpg` in the process's working directory.
7. **Shutdown**: pressing `q` exits the loop; `cap.release()` and `cv::destroyAllWindows()` release the camera and display resources.

## OpenCV API Used

- `cv::VideoCapture` opens a camera; `isOpened()` tests whether the device was opened; the stream extraction operator reads a frame; `release()` closes the capture device.
- `cv::Mat` stores the original, resized, and grayscale images.
- `cv::resize` resamples an image to a specified size. This application requests 640×360.
- `cv::Size` represents the target width and height passed to `cv::resize`.
- `cv::cvtColor` applies a color-space conversion. `cv::COLOR_BGR2GRAY` specifies conversion from OpenCV's BGR channel order to grayscale.
- `cv::Mat::rows`, `cols`, `channels()`, `total()`, and `type()` expose frame metadata; `at<cv::Vec3b>` accesses a BGR pixel and `at<unsigned char>` accesses one grayscale pixel.
- `cv::imshow` displays a frame, `cv::waitKey` handles GUI events and keyboard input, and `cv::destroyAllWindows` closes OpenCV windows.
- `cv::imwrite` encodes and saves the selected frame as an image file.
- `CV_VERSION` reports the OpenCV version used at compile time.

## Profiling and Bottleneck Notes

The code uses `std::chrono::steady_clock` and `std::chrono::duration<double, std::milli>` to measure elapsed wall-clock time. A monotonic clock is appropriate for durations because wall-clock adjustments do not affect it.

- **Combined capture/processing time** (`Avg Capture`): the timer starts immediately before `cap >> frame` and stops after `cv::cvtColor`. It currently includes camera read/wait time, resize, and grayscale conversion. It does **not** isolate camera acquisition or resize time.
- **Grayscale time** (`Avg Grayscale`): the inner timer measures only `cv::cvtColor` on the resized frame.
- **FPS**: `frameCount / elapsedSeconds` is calculated over roughly one-second reporting windows. This is the observed loop rate, including capture, processing, display/event handling, and scheduling; it is not a pure image-processing throughput measurement.

In the supplied run screenshot, the combined average is approximately **19–23 ms per frame** after the initial readings, grayscale conversion averages approximately **0.05–0.14 ms**, and the observed frame rate settles around **30 FPS**. Those observations indicate that grayscale conversion is not the dominant measured cost in that run. The combined timer cannot identify whether camera delivery or resizing accounts for most of the remaining time, so those are the next stages to time separately before naming a specific bottleneck. Results vary with the camera, resolution, machine, and runtime conditions.

The original 1280×720 frame contains 921,600 pixels. Resizing to 640×360 reduces this to 230,400 pixels, one quarter as many. The grayscale output remains 640×360 but uses one channel instead of three, reducing image data per pixel from three channel values to one. These steps lower the amount of image data subsequent stages need to handle.

## Captured Frames

### Original camera frame

![Original 1280 by 720 camera frame](assets/frames/captured_frame_original.jpg)

The unmodified camera frame is 1280×720 and has three BGR channels. It is the source image before the fixed-size resize and grayscale conversion.

### Resized color frame

![Resized 640 by 360 color frame](assets/frames/captured_frame_resizedframe.jpg)

The image is reduced from 1280×720 to 640×360 while retaining three BGR channels. Its pixel dimensions, and therefore pixel count, are one quarter of the original. The smaller frame is the input to the grayscale conversion.

### Grayscale frame

![Grayscale 640 by 360 frame](assets/frames/captured_frame_grayscale.jpg)

The frame remains 640×360, but BGR color has been converted to one grayscale channel using `cv::cvtColor` and `cv::COLOR_BGR2GRAY`. This is the image shown in the OpenCV window and saved when `s` is pressed.

## Profiling Run Screenshot

The terminal screenshot supplied with this project run shows the OpenCV version, successful camera startup, frame metadata, and repeated FPS and timing measurements described above. The screenshot is included in the conversation but is not available as a file in the project workspace, so it cannot be embedded in the repository yet. The three camera-frame images above are the captured files available locally.

## Build and Run

This project requires CMake, a C++17 compiler, and an OpenCV development installation with camera/video I/O support.

```sh
cmake -S . -B build
cmake --build build
./build/vision_app
```

When using VS Code with CMake Tools, configure the CMake project so the OpenCV include directories and libraries from `find_package(OpenCV REQUIRED)` are passed to the target.

In the OpenCV window, press `s` to save the displayed grayscale frame or `q` to quit. The camera must be accessible to the application; on macOS, grant camera permission to the terminal or application launching it.