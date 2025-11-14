#include <onnxruntime_cxx_api.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>

#define BG_GREEN 1
#define BG_BLUR 2
#define BG_IMG 3

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

inline void blend(const cv::Mat &bg, const cv::Mat &fg, const cv::Mat &mask, cv::Mat &res)
{
static cv::Mat mf,m3,f1,f2;

// normalize mask to 0-1 and convert to 3 channels
mask.convertTo(mf, CV_32FC1, 1.0 / 255.0);
cv::cvtColor(mf, m3, cv::COLOR_GRAY2BGR);

bg.convertTo(f1, CV_32FC1, 1.0 / 255.0);
fg.convertTo(f2, CV_32FC1, 1.0 / 255.0);

// blend with mask
cv::multiply(f1, cv::Scalar(1.0, 1.0, 1.0)-m3, f1);
cv::multiply(f2, m3, f2);

cv::add(f1, f2, res);
}

void blend_cuda(const cv::Mat &bg, const cv::Mat &fg, const cv::Mat &mask, cv::Mat &res)
{
cv::cuda::GpuMat g1, g2, gmask, gmask3, gout;
g1.upload(bg);
g2.upload(fg);
gmask.upload(mask);

g1.convertTo(g1, CV_32F, 1.0 / 255.0);
g2.convertTo(g2, CV_32F, 1.0 / 255.0);
gmask.convertTo(gmask, CV_32F, 1.0 / 255.0);

cv::cuda::GpuMat channels[3] = { gmask, gmask, gmask };
cv::cuda::merge(channels, 3, gmask3);

cv::cuda::GpuMat ginvMask(gmask3.size(), gmask3.type());
cv::cuda::subtract(cv::Scalar::all(1.0), gmask3, ginvMask);

// Multiply
cv::cuda::GpuMat p1, p2;
cv::cuda::multiply(g1, ginvMask, p1);
cv::cuda::multiply(g2, gmask3, p2);

// Add
cv::cuda::add(p1, p2, gout);

gout.download(res);
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

    cv::Size fsize=hdsize; // input and output size
    cv::Size size=sdsize; // size for rvm
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
    cap.set(cv::CAP_PROP_FRAME_WIDTH, fsize.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, fsize.height);
    cap.set(cv::CAP_PROP_FPS, 30);

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

    cv::Mat frame,iframe,bgimg,pmask;

    int bgtype=1;
    int run=1, green=1;
    bool show_img=true;
    bool show_mask=true;
    bool show_green=true;

    bgimg=cv::imread("bg.jpg", cv::IMREAD_COLOR);
    resize(bgimg, bgimg, fsize);

    const cv::Mat bg_green(fsize, CV_8UC3, cv::Scalar(0,255,0));
    cv::Mat bg(fsize, CV_8UC3);

    uint f=1;

    cv::TickMeter tm;

    while (run) {
	tm.start();

	cv::Mat frs;
        cap.read(iframe);
        if (iframe.empty()) {
            printf("error : empty frame grabbed");
            break;
        }
	f++;

	// Resize, potentially larger, input frame to smaller size for rvm
        resize(iframe, frame, size);

        cv::Mat blobMat;
        cv::dnn::blobFromImage(frame, blobMat, 1.0/255.0);

        src_data.assign(blobMat.begin<float>(), blobMat.end<float>());

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

        cv::Mat m(mask.size(), CV_8UC1, cv::Scalar(0));

#if 1
	m=mask;
#else
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

#endif
	// Resize mask input size
        resize(m, m, iframe.size());

#if 0
        cv::Mat bg(iframe.size(), CV_8UC3, cv::Scalar(0,255,0));
        cv::copyTo(iframe, bg, m);
        cv::imshow("green", bg);
#endif

        // Blur original frame
        cv::Mat b, bf;

	switch (bgtype) {
		case BG_GREEN:
	            bg=bg_green;
		break;
		case BG_BLUR:
	            cv::blur(iframe, bf, cv::Size(19,19));
		    bg=bf;
		break;
		case BG_IMG:
	            bg=bgimg;
		break;
	}

        if (show_green) {
            //b.convertTo(b, CV_8UC3, 255.0);
	    if (use_CUDA) {
  	       blend_cuda(bg, iframe, m, b);
	    } else {
  	       blend(bg, iframe, m, b);
	    }
            // cv::copyTo(frame, bg, m);
            cv::imshow("green", b);
        }

        if (show_img) {
	    cv::Mat img;
            cv::bitwise_and(iframe, iframe, img, m);
            cv::imshow("img", img);
        }

//	if (!pmask.empty())
//		cv::addWeighted(m, 0.9, pmask, 0.1, 0.0, m);

        if (show_mask)
            cv::imshow("mask", m);

//	pmask=m.clone();

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
            bgtype++;
	    if (bgtype>3) bgtype=1;
	    printf("Type: %d\n", bgtype);
            break;
        case 'n':
            bgtype--;
	    if (bgtype<1) bgtype=3;
	    printf("Type: %d\n", bgtype);
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
