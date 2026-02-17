import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

public class ColorDetectionHSV {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {

        VideoCapture camera = new VideoCapture(0);

        if (!camera.isOpened()) {
            System.out.println("Error: Camera not accessible");
            return;
        }

        System.out.println("Camera opened successfully");

        Mat frame = new Mat();
        Mat hsv = new Mat();
        Mat mask = new Mat();

        // Relaxed HSV range for BLUE (easy detection)
        Scalar lowerBlue = new Scalar(90, 50, 50);
        Scalar upperBlue = new Scalar(150, 255, 255);

        while (true) {

            camera.read(frame);
            if (frame.empty()) continue;

            // Convert BGR to HSV
            Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_BGR2HSV);

            // Create mask for blue color
            Core.inRange(hsv, lowerBlue, upperBlue, mask);

            // Show windows
            HighGui.imshow("Original", frame);
            HighGui.imshow("Mask", mask);

            // Exit on ESC
            if (HighGui.waitKey(1) >= 0) break;
        }

        camera.release();
        HighGui.destroyAllWindows();
        System.exit(0);
    }
}
