package org.jwyao.mnnpose;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.UseCase;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Size;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

import org.jwyao.mnnpose.mnn.MNNForwardType;


public class MainActivity extends AppCompatActivity {
    private TextView poseInfo;
    private ImageView resultImageView;

    private final AtomicBoolean detectFlag = new AtomicBoolean(false);
    // create an executor: use a single worker thread to ensure that tasks are executed in order
    ExecutorService xService = Executors.newSingleThreadExecutor();

    private long startTime = 0;
    private long endTime = 0;
    private String currentPose;
    private String imageName;
    protected Bitmap mutableBitmap;
    private int jsonCheck = -1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // save the state before the end of the Activity lifecycle
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        if (!hasPermission()) requestPermission();
        initView();
        // initialize PersonDetector and PoseEstimator
        PersonDetector.initMNN(2, MNNForwardType.FORWARD_CPU.type);
        PoseEstimator.initMNN(2, MNNForwardType.FORWARD_CPU.type);
        for (File exDir: getExternalFilesDirs(null)) {
            String configPath = exDir.getAbsolutePath().replace(
                    "Android/data/org.jwyao.mnnpose/files", "config.json");
            File jsonFile = new File(configPath);
            if (!jsonFile.exists()) continue;
            jsonCheck = PoseEstimator.preparePoseData(jsonFile);
        }
        if (jsonCheck < 0) currentPose = "配置文件有误!";
    }

    // check had permission
    private boolean hasPermission() {
        return checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED &&
                checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED &&
                checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
    }

    // request permission
    private void requestPermission() {
        requestPermissions(new String[]{Manifest.permission.CAMERA,
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
    }

    private void initView() {
        // show results on the screen
        poseInfo = findViewById(R.id.pose_info);
        resultImageView = findViewById(R.id.imageView);
        // open album
        Button selectImgBtn = findViewById(R.id.select_img_btn);
        selectImgBtn.setOnClickListener(v -> {
            detectFlag.set(false);
            CameraX.unbindAll();
            Intent intent = new Intent(Intent.ACTION_PICK);
            intent.setType("image/*");
            startActivityForResult(intent, 1);
        });
        // open camera
        Button openCamera = findViewById(R.id.open_camera);
        openCamera.setOnClickListener(v -> {
            detectFlag.set(false);
            startCamera();
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        String image_path;
        if (resultCode == Activity.RESULT_OK) {
            if (requestCode == 1 && data != null) {
                Uri image_uri = data.getData();
                image_path = getPathFromURI(MainActivity.this, image_uri);
                int startIndex = image_path.lastIndexOf("/");
                int endIndex = image_path.lastIndexOf(".");
                try {
                    FileInputStream fis = new FileInputStream(image_path);
                    imageName = image_path.substring(startIndex + 1,endIndex);
                    Bitmap bitmap = BitmapFactory.decodeStream(fis);
                    resultImageView.setImageBitmap(bitmap);
                    detectOnModel(bitmap);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static String getPathFromURI(Context context, Uri uri) {
        // get the path of the image
        String path;
        Cursor cursor = context.getContentResolver().query(uri, null, null, null, null);
        if (cursor == null) {
            path = uri.getPath();
        } else {
            cursor.moveToFirst();
            int idx = cursor.getColumnIndex(MediaStore.Images.ImageColumns.DATA);
            path = cursor.getString(idx);
            cursor.close();
        }
        return path;
    }

    private void startCamera() {
        CameraX.unbindAll();
        DetectAnalyzer detectAnalyzer = new DetectAnalyzer();
        imageName = "Camera";
        CameraX.bindToLifecycle(this, gainAnalyzer(detectAnalyzer));
    }

    private UseCase gainAnalyzer(DetectAnalyzer detectAnalyzer) {
        ImageAnalysisConfig.Builder analysisConfigBuilder = new ImageAnalysisConfig.Builder();
        // analysisConfigBuilder.setLensFacing(CameraX.LensFacing.BACK);
        analysisConfigBuilder.setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE);
        analysisConfigBuilder.setTargetResolution(new Size(1920, 1080)); // 1:1
        ImageAnalysisConfig config = analysisConfigBuilder.build();
        ImageAnalysis analysis = new ImageAnalysis(config);
        analysis.setAnalyzer(detectAnalyzer);
        return analysis;
    }

    private Bitmap imageToBitmap(ImageProxy image) {
        // YUV: mainly used in the video field(Y: brightness, UV: color)
        byte[] nv21 = imageToNV21(image);
        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 100, out);
        byte[] imageBytes = out.toByteArray();
        try {
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    private byte[] imageToNV21(ImageProxy image) {
        // NV21: YY_YY_VU_VU_VU
        ImageProxy.PlaneProxy[] planesYUV = image.getPlanes();
        ByteBuffer yBuffer = planesYUV[0].getBuffer();
        ByteBuffer uBuffer = planesYUV[1].getBuffer();
        ByteBuffer vBuffer = planesYUV[2].getBuffer();
        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();
        byte[] nv21 = new byte[ySize + uSize + vSize];
        // U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);
        return nv21;
    }

    private class DetectAnalyzer implements ImageAnalysis.Analyzer {
        @Override
        public void analyze(ImageProxy image, final int rotationDegrees) {
            final Bitmap bitmapSrc = imageToBitmap(image);
            detectOnModel(bitmapSrc);
        }
    }

    private void detectOnModel(Bitmap bitmapSrc) {
        if (detectFlag.get()) return;
        detectFlag.set(true);
        startTime = System.currentTimeMillis();
        xService.execute(() -> {
            detectAndDraw(bitmapSrc);
            showResultOnUI();
        });
    }

    protected void showResultOnUI() {
        runOnUiThread(() -> {
            detectFlag.set(false);
            resultImageView.setImageBitmap(mutableBitmap);
            endTime = System.currentTimeMillis();
            poseInfo.setText(String.format(Locale.CHINESE, "%s\n\nTime: %.3f s",
                    currentPose, (endTime - startTime) / 1000.0));
        });
    }

    protected void detectAndDraw(Bitmap image) {
        Bitmap biTmp = image;
        currentPose = "未检测到人体\n\n";
        float[] bBox = PersonDetector.detectPerson(image);
        if (bBox != null && jsonCheck > 0) {
            PoseEstimator.PersonPose ppResult = PoseEstimator.estimatePose(image, bBox);
            currentPose = String.format(Locale.CHINESE, "Category: %s\n\nScore: %.2f%%", ppResult.category, ppResult.score);
            biTmp = PoseEstimator.drawPersonPose(image, ppResult);
            String saveName = String.format(Locale.CHINESE, "%s_%s_%d", imageName, ppResult.category, Math.round(ppResult.score));
            saveBitmap(biTmp, saveName);
        }
        mutableBitmap = biTmp;
    }

    public static void saveBitmap(Bitmap bitmap,String position) {
        // save the detected image to SD card
        String savePath;
        if (Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED)) {
            String sdPath = Environment.getExternalStorageDirectory() + "/";
            savePath = sdPath + "Results/"; }
        else {
            Log.d("saveBitmap", "Error: sdcardPath");
            return;
        }
        try {
            File imageFile = new File(savePath + position + ".jpg");
            FileOutputStream imageFileOut = new FileOutputStream(imageFile);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, imageFileOut);
            imageFileOut.flush();
            imageFileOut.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onDestroy() {
        // stop the thread and release the memory
        detectFlag.set(false);
        if (xService != null) {
            xService.shutdown();
            xService = null;
        }
        CameraX.unbindAll();
        super.onDestroy();
    }

}
