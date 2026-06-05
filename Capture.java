// ================= IMPORT LIBRARIES =================

// OpenCV core packages
import org.opencv.core.*;

// Capture video frames from camera
import org.opencv.videoio.VideoCapture;

// Image processing functions
import org.opencv.imgproc.Imgproc;

// Face detection using Haar cascade
import org.opencv.objdetect.CascadeClassifier;

// Deep Learning model execution
import org.deeplearning4j.nn.graph.ComputationGraph;

// Import Keras-trained model
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;

// Multi-dimensional array for neural network
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

// GUI packages
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

// Collections for storing emotion history
import java.util.LinkedList;



// ================= MAIN CLASS =================

public class Capture extends JFrame {

    // Load OpenCV native library
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    // GUI Labels
    JLabel imageLabel;      // Displays camera frame
    JLabel emotionLabel;   // Displays detected emotion
    JLabel timerLabel;     // Displays countdown timer

    // Buttons
    JButton startButton;
    JButton stopButton;

    // Camera object
    VideoCapture camera;

    // Flag to control camera loop
    boolean running = false;

    // Emotion labels corresponding to model output
    static String[] emotions = {
            "Angry",
            "Disgust",
            "Fear",
            "Happy",
            "Sad",
            "Surprise",
            "Sad"
    };

    // Stores recent emotions (for smoothing if needed)
    static LinkedList<String> emotionHistory =
            new LinkedList<>();

    // Maximum stored history
    static final int HISTORY_SIZE = 10;

    // Tracks last update time
    static long lastUpdateTime = 0;

    // Countdown timer (seconds)
    int countdown = 5;

    // Deep learning model
    ComputationGraph model;

    // Face detector
    CascadeClassifier faceDetector;


    // ================= CONSTRUCTOR =================

    public Capture() {

        // Window title
        setTitle("Emotion Detection ");

        // Window size
        setSize(1000,750);

        // Layout manager
        setLayout(new BorderLayout());

        // Close program on exit
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // Background color
        getContentPane().setBackground(
                new Color(30,30,30)
        );

        // ===== TOP PANEL =====

        JPanel topPanel =
                new JPanel(new GridLayout(2,1));

        topPanel.setBackground(
                new Color(30,30,30)
        );

        // Emotion text label
        emotionLabel =
                new JLabel(
                        "Press Start",
                        SwingConstants.CENTER
                );

        emotionLabel.setFont(
                new Font("Segoe UI",
                        Font.BOLD,
                        36)
        );

        emotionLabel.setForeground(
                new Color(1,70,120)
        );

        // Countdown label
        timerLabel =
                new JLabel(
                        "Next update in: 5 sec",
                        SwingConstants.CENTER
                );

        timerLabel.setFont(
                new Font("Segoe UI",
                        Font.PLAIN,
                        18)
        );

        timerLabel.setForeground(Color.WHITE);

        // Add labels to panel
        topPanel.add(emotionLabel);
        topPanel.add(timerLabel);

        // Add panel to window
        add(topPanel, BorderLayout.NORTH);


        // ===== CAMERA DISPLAY PANEL =====

        imageLabel =
                new JLabel();

        imageLabel.setHorizontalAlignment(
                SwingConstants.CENTER
        ); // aligns to center

        add(imageLabel,
                BorderLayout.CENTER);


        // ===== BUTTON PANEL =====

        JPanel buttonPanel =
                new JPanel();

        buttonPanel.setBackground(
                new Color(30,30,30)
        );

        // Start button
        startButton =
                new JButton("Start");

        // Stop button
        stopButton =
                new JButton("Stop");

        // Style buttons
        styleButton(startButton,
                new Color(76,175,80));

        styleButton(stopButton,
                new Color(244,67,54));

        // Add buttons
        buttonPanel.add(startButton);
        buttonPanel.add(stopButton);

        add(buttonPanel,
                BorderLayout.SOUTH);

        // Make window visible
        setVisible(true);

        // Setup button actions
        setupActions();

        // Load AI models
        loadModels();

        // Start countdown timer
        startCountdownTimer();
    }


    // ================= BUTTON STYLING =================

    void styleButton(JButton button,
                     Color color) {

        // Set font
        button.setFont(
                new Font("Segoe UI",
                        Font.BOLD,
                        16)
        );

        // Set background color
        button.setBackground(color);

        // Text color
        button.setForeground(Color.WHITE);

        // Remove focus border
        button.setFocusPainted(false);

        // Button size
        button.setPreferredSize(
                new Dimension(140,45)
        );
    }


    // ================= LOAD MODELS =================

    void loadModels() {

        try {

            System.out.println(
                    "Loading emotion model..."
            );

            // Load trained Keras model
            model =
                    KerasModelImport.importKerasModelAndWeights(
                            "models/fer2013_mini_XCEPTION.102-0.66.hdf5"
                    );

            System.out.println(
                    "Emotion model loaded!"
            );

            // Load face detection model
            faceDetector =
                    new CascadeClassifier(
                            "haarcascade_frontalface_default.xml"
                    );

        }

        catch (Exception e) {

            e.printStackTrace();

        }
    }


    // ================= BUTTON ACTIONS =================

    void setupActions() {

        // Start camera
        startButton.addActionListener(
                e -> startCamera()
        );

        // Stop camera
        stopButton.addActionListener(
                e -> stopCamera()
        );
    }


    // ================= START CAMERA =================

    void startCamera() {

        // Prevent multiple starts
        if (running)
            return;

        // Open camera (0 = default webcam)
        camera =
                new VideoCapture(0);

        // Check if camera opened
        if (!camera.isOpened()) {

            System.out.println(
                    "Camera not detected!"
            );

            return;
        }

        running = true;

        // Start camera thread
        new Thread(() ->
                processCamera()
        ).start();
    }


    // ================= STOP CAMERA =================

    void stopCamera() {

        running = false;

        // Release camera resource
        if (camera != null)
            camera.release();

        emotionLabel.setText(
                "Camera Stopped"
        );
    }


    // ================= COUNTDOWN TIMER =================

    void startCountdownTimer() {

        new Timer(1000, e -> {

            // Decrease countdown
            if (countdown > 0) {

                countdown--;

                timerLabel.setText(
                        "Next update in: "
                        + countdown
                        + " sec"
                );

            }

        }).start();
    }


    // ================= CAMERA PROCESSING =================

    void processCamera() {

        Mat frame = new Mat();

        while (running) {

            // Read frame
            camera.read(frame);

            if (frame.empty())
                continue;

            // Convert to grayscale
            Mat gray = new Mat();

            Imgproc.cvtColor(
                    frame,
                    gray,
                    Imgproc.COLOR_BGR2GRAY
            );

            // Detect faces
            MatOfRect faces =
                    new MatOfRect();

            faceDetector.detectMultiScale(
                    gray,
                    faces,
                    1.1,
                    5,
                    0,
                    new Size(80,80),
                    new Size()
            );

            // Process each face
            for (Rect rect : faces.toArray()) {

                // Draw face rectangle
                Imgproc.rectangle(
                        frame,
                        rect.tl(),
                        rect.br(),
                        new Scalar(0,255,0),
                        2
                );

                // Crop face
                Mat face =
                        new Mat(gray, rect);

                // Resize to 48x48
                Imgproc.resize(
                        face,
                        face,
                        new Size(48,48)
                );

                // Improve contrast
                Imgproc.equalizeHist(face, face);

                // Create input tensor
                INDArray input =
                        Nd4j.create(1,48,48,1);

                // Normalize pixels
                for (int i=0;i<48;i++) {

                    for (int j=0;j<48;j++) {

                        double[] pixel =
                                face.get(i,j);

                        double normalized =
                                (pixel[0]-127.5)/127.5;

                        input.putScalar(
                                new int[]{0,i,j,0},
                                normalized
                        );
                    }
                }

                // Predict emotion
                INDArray output =
                        model.outputSingle(input);

                int emotionIndex =
                        Nd4j.argMax(
                                output,
                                1
                        ).getInt(0);

                String emotion =
                        emotions[emotionIndex];

                long currentTime =
                        System.currentTimeMillis();

                // Update every 5 seconds
                if (currentTime - lastUpdateTime
                        >= 5000) {

                    updateEmotionUI(emotion);

                    showEmotionPopup(emotion);

                    countdown = 5;

                    lastUpdateTime =
                            currentTime;
                }

                // Display emotion text on frame
                Imgproc.putText(
                        frame,
                        emotion,
                        new org.opencv.core.Point(
                                rect.x,
                                rect.y - 10
                        ),
                        Imgproc.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        new Scalar(0,255,0),
                        2
                );
            }

            // Convert frame to image
            BufferedImage image =
                    matToBufferedImage(frame);

            imageLabel.setIcon(
                    new ImageIcon(image)
            );

            repaint();
        }
    }


    // ================= UPDATE UI =================

    void updateEmotionUI(String emotion) {

        String emoji = "";

        switch (emotion) {

            case "Happy":
                emoji = " 😊";
                emotionLabel.setForeground(
                        new Color(76,175,80)
                );
                break;

            case "Sad":
                emoji = " 😢";
                emotionLabel.setForeground(
                        new Color(33,150,243)
                );
                break;

            case "Angry":
                emoji = " 😡";
                emotionLabel.setForeground(
                        new Color(244,67,54)
                );
                break;

            default:
                emoji = " 😮";
        }

        emotionLabel.setText(
                "User is "
                + emotion
                + emoji
        );
    }


    // ================= POPUP MESSAGE =================

    void showEmotionPopup(String emotion) {

        String message = "";

        switch (emotion) {

            case "Happy":
                message =
                        "Great! Keep smiling 😊";
                break;

            case "Sad":
                message =
                        "Be Happy! Stay Positive 😊";
                break;

            case "Angry":
                message =
                        "Relax! Take a deep breath 😌";
                break;

            case "Fear":
                message =
                        "Stay calm 👍";
                break;

            case "Surprise":
                message =
                        "Wow! That’s surprising 😮";
                break;

            case "Disgust":
                message =
                        "Try to stay positive 🙂";
                break;
        }

        JOptionPane.showMessageDialog(
                this,
                message,
                "Emotion Feedback",
                JOptionPane.INFORMATION_MESSAGE
        );
    }


    // ================= IMAGE CONVERSION =================

    public static BufferedImage matToBufferedImage(Mat mat) {

        int type =
                BufferedImage.TYPE_BYTE_GRAY;

        // If color image
        if (mat.channels() > 1)
            type =
                    BufferedImage.TYPE_3BYTE_BGR;

        int bufferSize =
                mat.channels()
                * mat.cols()
                * mat.rows();

        byte[] b =
                new byte[bufferSize];

        mat.get(0,0,b);

        BufferedImage image =
                new BufferedImage(
                        mat.cols(),
                        mat.rows(),
                        type
                );

        final byte[] targetPixels =
                ((java.awt.image.DataBufferByte)
                        image.getRaster()
                                .getDataBuffer())
                        .getData();

        System.arraycopy(
                b,
                0,
                targetPixels,
                0,
                b.length
        );

        return image;
    }


    // ================= MAIN METHOD =================

    public static void main(String[] args) {

        // Start application
        new Capture();

    }
}