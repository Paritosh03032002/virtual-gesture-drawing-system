import java.util.ArrayList;
import java.util.List;

import org.opencv.core.*;
import org.opencv.imgproc.Moments;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

public class AirCanvas {

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
        Mat canvas = new Mat();

        // 🔵 Safer blue range (works in most rooms)
        Scalar lowerBlue = new Scalar(90, 80, 60);
        Scalar upperBlue = new Scalar(130, 255, 255);

        Mat kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT, new Size(5, 5)
        );

        int prevX = -1, prevY = -1;

        while (true) {

            camera.read(frame);
            if (frame.empty()) continue;

            if (canvas.empty()) {
                canvas = Mat.zeros(frame.size(), CvType.CV_8UC3);
            }

            Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_BGR2HSV);
            Core.inRange(hsv, lowerBlue, upperBlue, mask);

            Imgproc.erode(mask, mask, kernel);
            Imgproc.dilate(mask, mask, kernel);

            List<MatOfPoint> contours = new ArrayList<>();
            Imgproc.findContours(mask, contours, new Mat(),
                    Imgproc.RETR_EXTERNAL,
                    Imgproc.CHAIN_APPROX_SIMPLE);

            double maxArea = 0;
            MatOfPoint bestContour = null;

            for (MatOfPoint c : contours) {
                double area = Imgproc.contourArea(c);
                if (area > 800 && area < 20000 && area > maxArea) {
                    maxArea = area;
                    bestContour = c;
                }
            }

            if (bestContour != null) {

                // Stable center point
                Moments m = Imgproc.moments(bestContour);
                if (m.m00 != 0) {

                    int cx = (int) (m.m10 / m.m00);
                    int cy = (int) (m.m01 / m.m00);

                    // Visual fingertip (top-most)
                    Point[] pts = bestContour.toArray();
                    Point tip = pts[0];
                    for (Point p : pts) {
                        if (p.y < tip.y) tip = p;
                    }

                    // Draw dot (use centroid for stability)
                    Imgproc.circle(frame,
                            new Point(cx, cy),
                            8,
                            new Scalar(0, 0, 255),
                            -1);

                    // Draw line using centroid (stable)
                    if (prevX != -1 && prevY != -1) {
                        Imgproc.line(canvas,
                                new Point(prevX, prevY),
                                new Point(cx, cy),
                                new Scalar(255, 0, 0),
                                5);
                    }

                    prevX = cx;
                    prevY = cy;
                }
            } else {
                prevX = -1;
                prevY = -1;
            }

            Core.add(frame, canvas, frame);

            HighGui.imshow("Air Canvas", frame);
            HighGui.imshow("Mask", mask);

            // 🔒 ESC logic unchanged
            if (HighGui.waitKey(1) >= 0) break;
        }

        camera.release();
        HighGui.destroyAllWindows();
        System.exit(0);
    }
}
