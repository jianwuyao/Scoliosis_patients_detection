package org.jwyao.mnnpose;

import android.os.Environment;
import android.graphics.Bitmap;
import android.graphics.Matrix;

import java.util.Arrays;
import java.util.Vector;
import java.lang.Math;

import org.jwyao.mnnpose.mnn.MNNImageProcess;
import org.jwyao.mnnpose.mnn.MNNNetInstance;

public class PersonDetector {
    private static MNNNetInstance.Session mSession = null;
    private static MNNNetInstance.Session.Tensor mInputTensor;

    public static void initMNN(int numThread, int forwardType) {
        MNNNetInstance.Config mConfig = new MNNNetInstance.Config();
        mConfig.numThread = numThread;
        mConfig.forwardType = forwardType;

        // prepare mnn net body_models
        String mModelPath = Environment.getExternalStorageDirectory().getPath() + "/MNN/body_ssd_300.mnn";
        // create net instance
        MNNNetInstance mNetInstance = MNNNetInstance.createFromFile(mModelPath);
        // mConfig.saveTensors;
        assert mNetInstance != null;
        mSession = mNetInstance.createSession(mConfig);
        mInputTensor = mSession.getInput("normalized_input_image_tensor");
    }

    public static Vector<float[]> detectPersons(Bitmap image) {
        // realize human body detection based on SSD
        int imageWidth = image.getWidth();
        int imageHeight = image.getHeight();
        final MNNImageProcess.Config config = new MNNImageProcess.Config();
        // normalization params
        config.mean = new float[] {0, 0, 0};
        config.normal = new float[] {1 / 255.f, 1 / 255.f, 1 / 255.f};
        config.source = MNNImageProcess.Format.BGRA;
        config.dest = MNNImageProcess.Format.RGB;
        // bitmap transform
        Matrix matrix = new Matrix();
        matrix.postScale((float) 300 / (float) imageWidth, 300 / (float) imageHeight);
        matrix.invert(matrix);
        // convert data to input tensor
        MNNImageProcess.convertBitmap(image, mInputTensor, config, matrix);

        // inference & get output tensor
        mSession.run(); // 19 * 19 * 3 + 10 * 10 * 6 + 5 * 5 * 6 + 3 * 3 * 6 + 2 * 2 * 6 + 1 * 1 * 6 = 1917
        MNNNetInstance.Session.Tensor tScores = mSession.getOutput("convert_scores"); // [1, 1917, 2]
        MNNNetInstance.Session.Tensor tBoxes = mSession.getOutput("Squeeze"); // [1917, 4]
        MNNNetInstance.Session.Tensor tAnchors = mSession.getOutput("anchors"); // [1917, 4]
        float[] scores = tScores.getFloatData(); // [1 * 1917 * 2]
        float[] boxes = tBoxes.getFloatData(); // [1917 * 4]
        float[] anchors = tAnchors.getFloatData();

        float X_SCALE = 10, Y_SCALE = 10, H_SCALE = 5, W_SCALE = 5;
        float score_threshold = 0.3f, nms_threshold = 0.45f;

        Vector<float[]> tmp_persons = new Vector<>();
        // postprocessing is not supported in MNN, so add decoding and NMS here
        for(int i = 0; i < 1917; i++) {
            // probability decoding, softmax
            float person_prob = scores[i * 2 + 1]; // nonperson_prob = scores[i * 2 + 0];
            if (person_prob < score_threshold) continue;

            // location decoding
            float y_center = boxes[i * 4    ] / Y_SCALE * anchors[i * 4 + 2] + anchors[i * 4    ];
            float x_center = boxes[i * 4 + 1] / X_SCALE * anchors[i * 4 + 3] + anchors[i * 4 + 1];
            float h = (float) (Math.exp(boxes[i * 4 + 2] / H_SCALE) * anchors[i * 4 + 2]);
            float w = (float) (Math.exp(boxes[i * 4 + 3] / W_SCALE) * anchors[i * 4 + 3]);
            float y_min = (y_center - h * 0.5f) * imageHeight;
            float x_min = (x_center - w * 0.5f) * imageWidth;
            float y_max = (y_center + h * 0.5f) * imageHeight;
            float x_max = (x_center + w * 0.5f) * imageWidth;

            x_min = x_min < imageWidth ? x_min > 0 ? x_min : 0 : imageWidth;
            y_min = y_min < imageHeight ? y_min > 0 ? y_min : 0 : imageHeight;
            x_max = x_max < imageWidth ? x_max > 0 ? x_max : 0 : imageWidth;
            y_max = y_max < imageHeight ? y_max > 0 ? y_max : 0 : imageHeight;
            tmp_persons.add(new float[] {x_min, y_min, x_max, y_max});
        }

        // NMS
        int N = tmp_persons.size();
        int[] labels = new int[N];
        Arrays.fill(labels, -1);
        for(int i = 0; i < N-1; i++) {
            for (int j = i+1; j < N; j++) {
                float[] pre_box = tmp_persons.get(i);
                float[] cur_box = tmp_persons.get(j);
                float iou_ = iou(pre_box, cur_box);
                if (iou_ > nms_threshold) labels[j] = 0;
            }
        }
        Vector<float[]> persons = new Vector<>();
        for (int i = 0; i < N; i++) {
            if (labels[i] == -1) persons.add(tmp_persons.get(i));
        }
        return persons;
    }

    public static float[] detectPerson(Bitmap image) {
        // from the multiple bodies detected, select the one with the largest area
        Vector<float[]> persons = detectPersons(image);
        int selected = -1;
        float diSMax = 0;
        for (int i = 0; i < persons.size(); i++) {
            float[] temP = persons.get(i);
            float diS = temP[2] - temP[0] + temP[3] - temP[1];
            if (diS > diSMax) {
                diSMax = diS;
                selected = i;
            }
        }
        return selected < 0 ? null : persons.get(selected);
    }

    protected static float iou(float[] box0, float[] box1) {
        // IOU
        float xMin0 = box0[0], yMin0 = box0[1], xMax0 = box0[2], yMax0 = box0[3];
        float xMin1 = box1[0], yMin1 = box1[1], xMax1 = box1[2], yMax1 = box1[3];
        float w = Math.max(0.0f, Math.min(xMax0, xMax1) - Math.max(xMin0, xMin1));
        float h = Math.max(0.0f, Math.min(yMax0, yMax1) - Math.max(yMin0, yMin1));
        float i = w * h, u = (xMax0 - xMin0) * (yMax0 - yMin0) + (xMax1 - xMin1) * (yMax1 - yMin1) - i;
        return u <= 0.0 ? 0.0f : i / u;
    }

}
