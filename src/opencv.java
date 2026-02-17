import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.videoio.VideoCapture;

public class opencv {

    static {
        // Load OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {

        // Open default webcam
        VideoCapture camera = new VideoCapture(0);

        // Check if camera opened successfully
        if (!camera.isOpened()) {
            System.out.println("Error: Camera not accessible");
            return;
        }

        System.out.println("Camera opened successfully");

        Mat frame = new Mat();

        while (true) {
            // Read frame from webcam
            camera.read(frame);

            // If frame is empty, skip this loop iteration
            if (frame.empty()) {
                System.out.println("Empty frame detected");
                continue;
            }

            // Display the webcam feed
            HighGui.imshow("Live Webcam", frame);

            // Exit when ESC key is pressed
            if (HighGui.waitKey(1) >= 0) {
                break;
            }
        }

        // Release camera and close windows
        camera.release();
        HighGui.destroyAllWindows();

        // Ensure JVM exits cleanly (IMPORTANT on Windows)
        System.exit(0);
    }
}
