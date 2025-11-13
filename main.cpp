#include <onnxruntime_cxx_api.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

int getMaxAreaContourId(std::vector <std::vector<cv::Point>> contours)
{
    double maxArea = 0;
    int maxAreaContourId = -1;
    for (int j = 0; j < contours.size(); j++) {
        double newArea = cv::contourArea(contours.at(j));
        if (newArea > maxArea) {
            maxArea = newArea;
            maxAreaContourId = j;
        }
    }
    return maxAreaContourId;
}


int main(int argc, char **argv)
{
    cv::Size sdsize = cv::Size(640, 360);
    cv::Size hdsize = cv::Size(1280, 720);
    cv::Size fhdsize = cv::Size(1920, 1080);

    const char *onnxModelFilename = argc >= 2 ? argv[1] : "rvm_mobilenetv3_fp32.onnx";
    bool use_CUDA = false;

    if (argc >= 3) {
        if (!strcmp(argv[2], "GPU") || !strcmp(argv[2], "CUDA"))
            use_CUDA = true;
        else if (!strcmp(argv[2], "CPU"))
            use_CUDA = false;
        else {
            printf("invalid argument 2 : %s, should be CPU, GPU or CUDA\n", argv[2]);
        }
    }
    int deviceID = 0;

    cv::Size size=sdsize;
    float dsratio=0.25f;

    //creates the onnx runtime environment
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "segmentation");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetExecutionMode(ORT_PARALLEL);
    sessionOptions.SetIntraOpNumThreads(6);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    //activates the CUDA backend
    if (use_CUDA) {
        OrtCUDAProviderOptions cuda_options;
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    }

    Ort::Session session(env, onnxModelFilename, sessionOptions);
    Ort::IoBinding io_binding(session);
    Ort::AllocatorWithDefaultOptions allocator;

    cv::VideoCapture cap;
    cap.open(deviceID, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        printf("can not open device %d\n", deviceID);
        return 0;
    }

    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, size.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, size.height);

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    //Not sure if this really allocate on the GPU, there is currently no documentation on it...
    Ort::MemoryInfo memoryInfoCuda("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);

    std::vector<float> src_data(size.width * size.height * 3);
    std::vector<int64_t> src_dims = {1, 3, size.height, size.width};
    Ort::Value src_tensor = Ort::Value::CreateTensor<float>(memoryInfo, src_data.data(), src_data.size(), src_dims.data(), 4);

    float downsample_ratio = dsratio;
    int64_t downsample_ratio_dims[] = {1};
    Ort::Value downsample_ratio_tensor = Ort::Value::CreateTensor<float>(memoryInfo, &downsample_ratio, 1, downsample_ratio_dims, 1);

    float rec_data = 0.0f;
    int64_t rec_dims[] = {1, 1, 1, 1};
    Ort::Value r1i = Ort::Value::CreateTensor<float>(memoryInfo, &rec_data, 1, rec_dims, 4);

    io_binding.BindOutput("fgr", memoryInfoCuda);
    io_binding.BindOutput("pha", memoryInfo);
    io_binding.BindOutput("r1o", memoryInfoCuda);
    io_binding.BindOutput("r2o", memoryInfoCuda);
    io_binding.BindOutput("r3o", memoryInfoCuda);
    io_binding.BindOutput("r4o", memoryInfoCuda);

    io_binding.BindInput("r1i", r1i);
    io_binding.BindInput("r2i", r1i);
    io_binding.BindInput("r3i", r1i);
    io_binding.BindInput("r4i", r1i);
    io_binding.BindInput("downsample_ratio", downsample_ratio_tensor);

    cv::namedWindow("mask", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO );
    cv::namedWindow("img", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO );
    cv::namedWindow("green", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO );

    cv::Mat frame,bgimg,pmask;

    int run=1, green=1;
    bool show_img=true;
    bool show_mask=true;
    bool show_green=true;
    bool blurbg=false;

    bgimg=cv::imread("bg.jpg", cv::IMREAD_COLOR);
    resize(bgimg, bgimg, size);

    uint f=1;

    cv::TickMeter tm;

    while (run) {
	tm.start();

	cv::Mat frs;
        cap.read(frame);
        if (frame.empty()) {
            printf("error : empty frame grabbed");
            break;
        }
	f++;

        resize(frame, frame, size);

        cv::Mat blobMat;
        cv::dnn::blobFromImage(frame, blobMat, 1.0/255.0);

        src_data.assign(blobMat.begin<float>(), blobMat.end<float>());
//        for(size_t i = 0; i < src_data.size(); i++)
 //           src_data[i] /= 255;

        io_binding.BindInput("src", src_tensor);
        session.Run(Ort::RunOptions{nullptr}, io_binding);
        
        std::vector<std::string> outputNames = io_binding.GetOutputNames();
        std::vector<Ort::Value> outputValues = io_binding.GetOutputValues();
        
        cv::Mat mask(size.height, size.width, CV_8UC1);
        for (int i = 0; i < outputNames.size(); i++) {
            if (outputNames[i] == "pha") {
                const cv::Mat outputImg(size.height, size.width, CV_32FC1, const_cast<float*>(outputValues[i].GetTensorData<float>()));
                outputImg.convertTo(mask, CV_8UC1, 255.0);
            } else if (outputNames[i] == "r1o") {
                io_binding.BindInput("r1i", outputValues[i]);
            } else if (outputNames[i] == "r2o") {
                io_binding.BindInput("r2i", outputValues[i]);
            } else if (outputNames[i] == "r3o") {
                io_binding.BindInput("r3i", outputValues[i]);
            } else if (outputNames[i] == "r4o") {
                io_binding.BindInput("r4i", outputValues[i]);
            }
        }

//        cv::imshow("rawmask", mask);

        cv::Mat img;

        std::vector<std::vector<cv::Point> > cvs;
        cv::findContours(mask, cvs, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

        cv::Mat m(mask.size(), CV_8UC1, cv::Scalar(0));
#if 0
        for (int ci=0; ci<cvs.size();ci++) {
            cv::drawContours(m, cvs, ci, cv::Scalar(255), -1);
        }
#else
	int ci=getMaxAreaContourId(cvs);
	if (ci>-1) {
            cv::drawContours(m, cvs, ci, cv::Scalar(255), -1, cv::LINE_AA);
            cv::blur(m, m, cv::Size(9, 9));
	}
#endif

#if 0
        cv::Mat bg(frame.size(), CV_8UC3, cv::Scalar(0,255,0));
        cv::copyTo(frame, bg, m);
        cv::imshow("green", bg);
#endif

        // Blur original frame
        cv::Mat f1,f2, b, m3, mf;
        cv::Mat bg(frame.size(), CV_8UC3);

        if (green) {
            bg=cv::Scalar(0,255,0);
        } else if (blurbg) {
            cv::blur(frame, bg, cv::Size(19,19));
        } else {
            bg=bgimg;
        }

        if (show_green) {
            // normalize mask to 0-1 and convert to 3 channels
            m.convertTo(mf, CV_32FC1, 1.0 / 255.0);
            cv::cvtColor(mf, m3, cv::COLOR_GRAY2BGR);

            bg.convertTo(f1, CV_32FC1, 1.0 / 255.0);
            frame.convertTo(f2, CV_32FC1, 1.0 / 255.0);

            // blend with mask
            cv::multiply(f1, cv::Scalar(1.0, 1.0, 1.0)-m3, f1);
            cv::multiply(f2, m3, f2);

            cv::add(f1, f2, b);
            b.convertTo(b, CV_8UC3, 255.0);

            // cv::copyTo(frame, bg, m);
            cv::imshow("green", b);
        }

        if (show_img) {
            cv::bitwise_and(frame, frame, img, m);
            cv::imshow("img", img);
        }

	if (!pmask.empty())
		cv::addWeighted(m, 0.9, pmask, 0.1, 0.0, m);

        if (show_mask)
            cv::imshow("mask", m);

	pmask=m.clone();

	tm.stop();
	if (f % 32==0) {
		printf("FPS: %f (%f)\n", tm.getFPS(), tm.getAvgTimeMilli());
	}

        int key = cv::waitKey(1);
        switch (key) {
        case 'q':
            run=false;
            break;
        case 'm':
            show_mask=!show_mask;
            break;
        case 'i':
            show_img=!show_img;
            break;
        case 'g':
            show_green=!show_green;
            break;
        case 'b':
            green=!green;
            break;
	case 'f':
	    cv::setWindowProperty("mask", cv::WND_PROP_FULLSCREEN , cv::WINDOW_FULLSCREEN );
	break;
	case 'r':
	    cv::setWindowProperty("mask", cv::WND_PROP_FULLSCREEN , cv::WINDOW_NORMAL);
	break;
	case 'd':
	    cv::setWindowProperty("green", cv::WND_PROP_FULLSCREEN , cv::WINDOW_FULLSCREEN );
	break;
	case 'e':
	    cv::setWindowProperty("green", cv::WND_PROP_FULLSCREEN , cv::WINDOW_NORMAL);
	break;
        }


    }
    return 0;
}
