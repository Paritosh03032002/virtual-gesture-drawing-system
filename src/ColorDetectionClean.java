import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

public class ColorDetectionClean {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {

        VideoCapture camera = new VideoCapture(0);

        if (!camera.isOpened()) {
            System.out.println("Camera not accessible");
            return;
        }

        Mat frame = new Mat();
        Mat hsv = new Mat();
        Mat mask = new Mat();

        // HSV range for blue
        Scalar lowerBlue = new Scalar(90, 50, 50);
        Scalar upperBlue = new Scalar(150, 255, 255);

        // Kernel for noise removal
        Mat kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT, new Size(5, 5)
        );

        while (true) {

            camera.read(frame);
            if (frame.empty()) continue;

            // Convert to HSV
            Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_BGR2HSV);

            // Color detection
            Core.inRange(hsv, lowerBlue, upperBlue, mask);

            // Noise removal
            Imgproc.erode(mask, mask, kernel);
            Imgproc.dilate(mask, mask, kernel);

            HighGui.imshow("Original", frame);
            HighGui.imshow("Clean Mask", mask);

            // Exit on any key
            if (HighGui.waitKey(1) >= 0) break;
        }

        camera.release();
        HighGui.destroyAllWindows();
        System.exit(0);
    }
}
