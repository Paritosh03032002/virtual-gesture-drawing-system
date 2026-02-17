import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.videoio.VideoCapture;

public class AirCanvasDay9 {

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

        // Strict blue marker
        Scalar lowerBlue = new Scalar(105, 150, 80);
        Scalar upperBlue = new Scalar(125, 255, 255);

        Mat kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT, new Size(5, 5)
        );

        int prevX = -1, prevY = -1;
        boolean drawingEnabled = true;

        // For smoothing
        Deque<Point> pointBuffer = new ArrayDeque<>();
        int SMOOTHING_WINDOW = 5;

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

                Moments m = Imgproc.moments(bestContour);
                if (m.m00 != 0) {

                    int cx = (int) (m.m10 / m.m00);
                    int cy = (int) (m.m01 / m.m00);

                    int height = frame.rows();

                    // Top zone → Pause / Resume
                    if (cy < height * 0.15) {
                        drawingEnabled = !drawingEnabled;
                        prevX = -1;
                        prevY = -1;
                        pointBuffer.clear();
                        try { Thread.sleep(300); } catch (Exception e) {}
                    }

                    // Bottom zone → Clear canvas
                    else if (cy > height * 0.85) {
                        canvas.setTo(new Scalar(0, 0, 0));
                        prevX = -1;
                        prevY = -1;
                        pointBuffer.clear();
                        try { Thread.sleep(300); } catch (Exception e) {}
                    }

                    // Middle zone → Draw
                    else if (drawingEnabled) {

                        // Add to smoothing buffer
                        pointBuffer.add(new Point(cx, cy));
                        if (pointBuffer.size() > SMOOTHING_WINDOW) {
                            pointBuffer.poll();
                        }

                        // Compute average point
                        int sumX = 0, sumY = 0;
                        for (Point p : pointBuffer) {
                            sumX += p.x;
                            sumY += p.y;
                        }

                        int sx = sumX / pointBuffer.size();
                        int sy = sumY / pointBuffer.size();

                        Imgproc.circle(frame,
                                new Point(sx, sy),
                                8,
                                new Scalar(0, 0, 255),
                                -1);

                        if (prevX != -1 && prevY != -1) {

                            // Dynamic thickness
                            double dist = Math.hypot(sx - prevX, sy - prevY);
                            int thickness = (int) Math.max(2, 10 - dist / 5);

                            Imgproc.line(canvas,
                                    new Point(prevX, prevY),
                                    new Point(sx, sy),
                                    new Scalar(255, 0, 0),
                                    thickness);
                        }

                        prevX = sx;
                        prevY = sy;
                    }
                }
            } else {
                prevX = -1;
                prevY = -1;
                pointBuffer.clear();
            }

            // Status text
            String status = drawingEnabled ? "DRAWING" : "PAUSED";
            Imgproc.putText(frame,
                    status,
                    new Point(20, 40),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    1,
                    drawingEnabled ? new Scalar(0,255,0) : new Scalar(0,0,255),
                    2);

            Core.add(frame, canvas, frame);

            HighGui.imshow("Air Canvas - Day 9", frame);
            HighGui.imshow("Mask", mask);

            // ESC logic unchanged
            if (HighGui.waitKey(1) >= 0) break;
        }

        camera.release();
        HighGui.destroyAllWindows();
        System.exit(0);
    }
}
