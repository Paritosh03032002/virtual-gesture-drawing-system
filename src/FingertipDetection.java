import java.util.ArrayList;
import java.util.List;

import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;q
import org.opencv.imgproc.Moments;

public class FingertipDetection {

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

        Scalar lowerBlue = new Scalar(90, 50, 50);
        Scalar upperBlue = new Scalar(150, 255, 255);

        Mat kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT, new Size(5, 5)
        );

        while (true) {

            camera.read(frame);
            if (frame.empty()) continue;

            Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_BGR2HSV);
            Core.inRange(hsv, lowerBlue, upperBlue, mask);

            Imgproc.erode(mask, mask, kernel);
            Imgproc.dilate(mask, mask, kernel);

            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();

            Imgproc.findContours(
                    mask,
                    contours,
                    hierarchy,
                    Imgproc.RETR_EXTERNAL,
                    Imgproc.CHAIN_APPROX_SIMPLE
            );

            if (!contours.isEmpty()) {

                // Find largest contour
                double maxArea = 0;
                MatOfPoint largestContour = null;

                for (MatOfPoint c : contours) {
                    double area = Imgproc.contourArea(c);
                    if (area > maxArea) {
                        maxArea = area;
                        largestContour = c;
                    }
                }

                if (largestContour != null) {

                    Moments m = Imgproc.moments(largestContour);

                    if (m.m00 != 0) {
                        int cx = (int) (m.m10 / m.m00);
                        int cy = (int) (m.m01 / m.m00);

                        // Draw circle at fingertip
                        Imgproc.circle(
                                frame,
                                new Point(cx, cy),
                                10,
                                new Scalar(0, 0, 255),
                                -1
                        );

                        Imgproc.putText(
                                frame,
                                "(" + cx + ", " + cy + ")",
                                new Point(cx + 10, cy - 10),
                                Imgproc.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                new Scalar(0, 255, 0),
                                1
                        );
                    }
                }
            }

            HighGui.imshow("Fingertip Detection", frame);
            HighGui.imshow("Mask", mask);

            // Exit on any key
            if (HighGui.waitKey(1) >= 0) break;
        }

        camera.release();
        HighGui.destroyAllWindows();
        System.exit(0);
    }
}
