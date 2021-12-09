package org.jwyao.mnnpose;

import android.os.Environment;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.util.Log;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.Arrays;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import org.jwyao.mnnpose.mnn.MNNImageProcess;
import org.jwyao.mnnpose.mnn.MNNNetInstance;

public class PoseEstimator {
    private static MNNNetInstance.Session mSession = null;
    private static MNNNetInstance.Session.Tensor mInputTensor;

    public static void initMNN(int numThread, int forwardType) {
        MNNNetInstance.Config mConfig = new MNNNetInstance.Config();
        mConfig.numThread = numThread;
        mConfig.forwardType = forwardType;

        // prepare mnn net pose_models
        String mModelPath = Environment.getExternalStorageDirectory().getPath() + "/MNN/pose_hrnet_256.mnn";
        // create net instance
        MNNNetInstance mNetInstance = MNNNetInstance.createFromFile(mModelPath);
        // mConfig.saveTensors;
        assert mNetInstance != null;
        mSession = mNetInstance.createSession(mConfig);
        mInputTensor = mSession.getInput("box_detection");
    }

    private static String[] thCategory;
    private static int[] thScore;
    private static int[] thIndex_w;
    private static int[] thIndex_t;

    public static int preparePoseData(File jsonFile) {
        try {
            InputStreamReader inputStreamReader = new InputStreamReader(new FileInputStream(jsonFile));
            BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
            String line;
            StringBuilder jsonString = new StringBuilder();
            while ((line = bufferedReader.readLine()) != null) {
                jsonString.append(line);
            }
            bufferedReader.close();
            inputStreamReader.close();

            JSONObject jsonData = new JSONObject(jsonString.toString());
            JSONArray category = jsonData.getJSONArray("category");
            JSONArray score_threshold = jsonData.getJSONArray("score_threshold");
            JSONArray index_weight = jsonData.getJSONArray("index_weight");
            JSONArray index_threshold = jsonData.getJSONArray("index_threshold");

            int sum_weight = 0;
            int total_category = category.length(), total_score = score_threshold.length();
            int total_index_w = index_weight.length(), total_index_t = index_threshold.length();
            thCategory = new String[total_category];
            thScore = new int[total_score];
            thIndex_w = new int[total_index_w];
            thIndex_t = new int[total_index_t];
            for (int i = 0; i < total_category; i++) thCategory[i] = category.getString(i);
            for (int j = 0; j < total_score; j++) thScore[j] = score_threshold.getInt(j);
            for (int k = 0; k < total_index_w; k++) {
                thIndex_w[k] = index_weight.getInt(k);
                sum_weight += thIndex_w[k];
            }
            for (int l = 0; l < total_index_t; l++) thIndex_t[l] = index_threshold.getInt(l);

            if (total_index_w != total_index_t || sum_weight != 100) return -1;
            return 1;
        } catch (IOException | JSONException e) {
            e.printStackTrace();
            return -1;
        }
    }


    public static class PersonPose {
        public float[] x;
        public float[] y;
        public float[] s;
        public float[] box;
        public String category;
        public double score;

        public PersonPose() {
            this.x = new float[16];
            this.y = new float[16];
            this.s = new float[16];
            this.box = new float[] {0, 0, 0, 0};
            this.category = "";
            this.score = 0;
        }
    }

    private static final PersonPose personPose = new PersonPose();

    public static PersonPose estimatePose(Bitmap image, float[] roi) {
        // realize human pose estimation based on HRNet
        personPose.box = roi;
        int xMin = (int)roi[0], yMin = (int)roi[1], xMax = (int)roi[2], yMax = (int)roi[3];
        int roiWidth = xMax - xMin, roiHeight = yMax - yMin;
        Bitmap roiImage = Bitmap.createBitmap(image, xMin, yMin, roiWidth, roiHeight, null, false);
        int dstSize = Math.max(roiWidth, roiHeight);
        // make the image on the canvas, not change the image ratio
        Bitmap blackCanvas = Bitmap.createBitmap(dstSize, dstSize, Bitmap.Config.ARGB_8888);
        Bitmap mutableCanvas = blackCanvas.copy(blackCanvas.getConfig(), true);
        Canvas canvas = new Canvas(mutableCanvas);
        canvas.drawBitmap(roiImage, dstSize/2.0f - roiWidth/2.0f,
                dstSize/2.0f - roiHeight/2.0f, new Paint());
        // normalization params
        final MNNImageProcess.Config config = new MNNImageProcess.Config();
        config.mean = new float[] {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
        config.normal = new float[] {1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f};
        config.source = MNNImageProcess.Format.BGRA;
        config.dest = MNNImageProcess.Format.RGB;
        // bitmap transform
        Matrix matrix = new Matrix();
        matrix.postScale(256 / (float) dstSize, 256 / (float) dstSize);
        matrix.invert(matrix);
        // convert data to input tensor
        MNNImageProcess.convertBitmap(mutableCanvas, mInputTensor, config, matrix);

        // inference & get output tensor
        mSession.run();
        MNNNetInstance.Session.Tensor output = mSession.getOutput("pose_estimation"); // [1, 16, 64, 64]
        float[] hmData = output.getFloatData(); // [1 * 16 * 64 * 64]
        int hmWidth = 64, hmHeight = 64;
        // postprocessing: get max_heat location
        for (int i = 0; i < 16; i++) {
            float max_heat = 0, max_x = 0, max_y = 0;
            for (int r = 0; r < hmHeight; r++) {
                for (int c = 0; c < hmWidth; c++) {
                    float heat = hmData[(i * hmHeight + r) * hmWidth + c];
                    if (heat > max_heat) {
                        max_x = c;
                        max_y = r;
                        max_heat = heat;
                    }
                }
            }
            // eliminate the effects of rounding up and down
            int px = (int)Math.floor(max_x + 0.5f);
            int py = (int)Math.floor(max_y + 0.5f);
            if ((1 < px & px < hmWidth - 1) & (1 < py & py < hmHeight - 1)) {
                float heat_right = hmData[(i * hmHeight + py) * hmWidth + px + 1];
                float heat_left = hmData[(i * hmHeight + py) * hmWidth + px - 1];
                float heat_up = hmData[(i * hmHeight + py + 1) * hmWidth + px];
                float heat_down = hmData[(i * hmHeight + py - 1) * hmWidth + px];
                if (heat_right > heat_left) max_x += 0.25f;
                else if (heat_right < heat_left) max_x -= 0.25f;
                if (heat_up > heat_down) max_y += 0.25f;
                else if (heat_up < heat_down) max_y -= 0.25f;
            }
            personPose.x[i] = max_x * dstSize / hmWidth - (dstSize/2.0f - roiWidth/2.0f) + xMin;
            personPose.y[i] = max_y * dstSize / hmHeight - (dstSize/2.0f - roiHeight/2.0f)  + yMin;
            personPose.s[i] = max_heat;
        }

        classifyPose(personPose.box, personPose.x, personPose.y);
        return personPose;
    }

    protected static void classifyPose(float[] roi, float[] pX, float[] pY) {
        // define a variety of evaluation criterion
        // because image sizes are not uniform, criterion needs to be normalized
        int roiWidth = (int)(roi[2] - roi[0]), roiHeight = (int)(roi[3] - roi[1]);
        double[] criterion = new double[thIndex_w.length];
        criterion[0] = (Math.abs((pY[2] - pY[3])) / roiHeight * 1000) < thIndex_t[0] ?
                (Math.abs((pY[2] - pY[3])) / roiHeight * 1000) / thIndex_t[0] : 1;
        criterion[1] = (Math.abs((pY[12] - pY[13])) / roiHeight * 1000) < thIndex_t[1] ?
                (Math.abs((pY[12] - pY[13])) / roiHeight * 1000) / thIndex_t[1] : 1;
        criterion[2] = (Math.abs(pX[6] - (pX[2] + pX[3])/2.0f) / roiWidth * 1000) < thIndex_t[2] ?
                (Math.abs(pX[6] - (pX[2] + pX[3])/2.0f) / roiWidth * 1000) / thIndex_t[2] : 1;
        criterion[3] = (Math.abs(pX[7] - (pX[12] + pX[13])/2.0f) / roiWidth * 1000) < thIndex_t[3] ?
                (Math.abs(pX[7] - (pX[12] + pX[13])/2.0f) / roiWidth * 1000) / thIndex_t[3] : 1;
        criterion[4] = (Math.abs(Math.abs(pX[2] - pX[12]) - Math.abs(pX[3] - pX[13])) / roiWidth * 100) < thIndex_t[4] ?
                (Math.abs(Math.abs(pX[2] - pX[12]) - Math.abs(pX[3] - pX[13])) / roiWidth * 100) / thIndex_t[4] : 1;
        criterion[5] = (Math.abs(Math.abs(pY[2] - pY[12]) - Math.abs(pY[3] - pY[13])) / roiHeight * 1000) < thIndex_t[5] ?
                (Math.abs(Math.abs(pY[2] - pY[12]) - Math.abs(pY[3] - pY[13])) / roiHeight * 1000) / thIndex_t[5] : 1;

        float x2_11 = pX[2] - pX[11], y2_11 = pY[2] - pY[11], x3_14 = pX[3] - pX[14], y3_14 = pY[3] - pY[14];
        double dist2_11 = Math.sqrt(x2_11 * x2_11 + y2_11 * y2_11), dist3_14 = Math.sqrt(x3_14 * x3_14 + y3_14 * y3_14);
        double normDist = Math.abs(dist2_11 - dist3_14) / Math.sqrt(roiWidth * roiWidth + roiHeight * roiHeight) * 1000;
        criterion[6] = normDist < thIndex_t[6] ? normDist / thIndex_t[6] : 1;

        float x6_7 = pX[6] - pX[7], y6_7 = pY[6] - pY[7], x7_8 = pX[7] - pX[8], y7_8 = pY[7] - pY[8];
        criterion[7] = calcAngle(x6_7, y6_7, x7_8, y7_8);

        double finalScore = 0;
        for (int m = 0; m < thIndex_w.length; m++) {
            finalScore += criterion[m] * thIndex_w[m];
        }
        personPose.score = finalScore;
        personPose.category = thCategory[thScore.length];
        for (int n = 0; n < thScore.length; n++) {
            if (finalScore <= thScore[n]) {
                personPose.category = thCategory[n];
                break;
            }
        }
    }

    protected static double calcAngle(float x0, float y0, float x1, float y1) {
        // vector(x) * vector(y) = /x/ * /y/ * cos(theta)
        double delta = thIndex_t[7] * Math.PI / 180;
        double l0xl1 = Math.sqrt(x0 * x0 + y0 * y0) * Math.sqrt(x1 * x1 + y1 * y1);
        double theta = Math.acos((x0 * x1 + y0 * y1) / l0xl1);
        return theta < Math.PI / 2 ? theta < delta ? 1 : Math.pow(1 - 2 * (theta - delta) / (Math.PI - 2 * delta), 3) : 0;
    }

    private static final float minConf = 0.25f;

    static Bitmap drawPersonPose(Bitmap immutableBitmap, PersonPose personPose) {
        Bitmap mutableBitmap = immutableBitmap.copy(immutableBitmap.getConfig(), true);
        Canvas canvas = new Canvas(mutableBitmap);
        final Paint posePaint = new Paint();
        posePaint.setStyle(Paint.Style.STROKE);

        // 0:right_ankle,  1:right_knee,  2:right_hip,  3:left_hip,  4:left_knee,  5:left_ankle,
        // 6:pelvis,  7:thorax,  8:upper neck,  9:head top,  10:right_wrist,  11:right_elbow,
        // 12:right_shoulder,  13:left_shoulder,  14:left_elbow,  15:left_wrist
        // Color: 0xFF means 100% transparency
        int[][] jointPairs = {{0, 1, 0xFFFF4500}, {1, 2, 0xFFFFA500}, {2, 6, 0xFFFFFF00}, {6, 3, 0xFFFFFF00},
                {3, 4, 0xFFFFA500}, {4, 5, 0xFFFF4500}, {6, 7, 0xFF00FF00}, {7, 8, 0xFF32CD32},
                {8, 9, 0xFF008000}, {7, 12, 0xFFFFFF00}, {12, 11, 0xFFFFA500}, {11, 10, 0xFFFF4500},
                {7, 13, 0xFFFFFF00}, {13, 14, 0xFFFFA500}, {14, 15, 0xFFFF4500}};

        float[] kpX = Arrays.copyOf(personPose.x, 16);
        float[] kpY = Arrays.copyOf(personPose.y, 16);
        float[] kpS = Arrays.copyOf(personPose.s, 16);
        // draw limb
        posePaint.setStrokeWidth(8 * mutableBitmap.getWidth() / 800.0f);
        for (int[] pair : jointPairs) {
            if (kpS[pair[0]] < minConf || kpS[pair[1]] < minConf) continue;
            posePaint.setColor(pair[2]);
            canvas.drawLine(kpX[pair[0]], kpY[pair[0]], kpX[pair[1]], kpY[pair[1]], posePaint);
        }
        // draw keypoint
        posePaint.setStrokeWidth(16 * mutableBitmap.getWidth() / 800.0f);
        for (int n = 0; n < 16; n++) {
            if (kpS[n] < minConf) continue;
            posePaint.setColor(Color.RED);
            canvas.drawPoint(kpX[n], kpY[n], posePaint);
        }
        // draw box
        posePaint.setColor(Color.WHITE);
        posePaint.setStrokeWidth(4 * mutableBitmap.getWidth() / 800.0f);
        canvas.drawRect(personPose.box[0], personPose.box[1], personPose.box[2], personPose.box[3], posePaint);

        return mutableBitmap;
    }

}
