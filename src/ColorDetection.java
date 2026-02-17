import org.opencv.core.*;
import org.opencv.videoio.Videoio;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

public class ColorDetection {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {

        VideoCapture camera = new VideoCapture(0, Videoio.CAP_DSHOW);

        if (!camera.isOpened()) {
            System.out.println("Camera not available");
            return;
        }

        Mat frame = new Mat();
        Mat hsv = new Mat();
        Mat mask = new Mat();

        // HSV range for BLUE color (you can change later)
        Scalar lowerBlue = new Scalar(100, 150, 50);
        Scalar upperBlue = new Scalar(140, 255, 255);

        while (true) {
            camera.read(frame);
            if (frame.empty()) break;

            // Convert BGR to HSV
            Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_BGR2HSV);

            // Detect blue color
            Core.inRange(hsv, lowerBlue, upperBlue, mask);

            // Show original and mask
            HighGui.imshow("Original", frame);
            HighGui.imshow("Mask", mask);

            if (HighGui.waitKey(1) >= 0) break; // ESC
        }

        camera.release();
        HighGui.destroyAllWindows();
    }
}
