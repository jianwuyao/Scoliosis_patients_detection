package org.jwyao.mnnpose.mnn;

import android.graphics.Bitmap;
import android.graphics.Matrix;

public class MNNImageProcess {

    public enum Format {
        RGBA(0),
        RGB(1),
        BGR(2),
        GRAY(3),
        BGRA(4),
        YUV_420(10),
        YUV_NV21(11);

        public int type;

        Format(int t) {
            type = t;
        }
    }

    public enum Filter {
        NEAREST(0),
        BILINEAL(1),
        BICUBIC(2);

        public int type;

        Filter(int t) {
            type = t;
        }
    }

    public enum Wrap {
        CLAMP_TO_EDGE(0),
        ZERO(1),
        REPEAT(2);

        public int type;

        Wrap(int t) {
            type = t;
        }
    }


    public static class Config {
        // default
        public float[] mean = {0f, 0f, 0f, 0f};
        public float[] normal = {1f, 1f, 1f, 1f};
        public Format source = Format.RGBA;
        public Format dest = Format.BGR;
        public Filter filter = Filter.NEAREST;
        public Wrap wrap = Wrap.CLAMP_TO_EDGE;
    }

    /**
     * 设置输入buffer
     *
     * @param buffer 输入的buffer
     * @param tensor 输入的Tensor
     * @param config 配置mean、normal、图片目标格式
     * @param matrix 定义裁剪、缩放、旋转等
     * @return boolean
     */
    public static boolean convertBuffer(byte[] buffer, int width, int height, MNNNetInstance.Session.Tensor tensor, Config config, Matrix matrix) {
        if (matrix == null) {
            matrix = new Matrix();
        }
        float[] value = new float[9];
        matrix.getValues(value);

        return MNNNetNative.nativeConvertBufferToTensor(buffer, width, height, tensor.instance(),
                config.source.type, config.dest.type, config.filter.type, config.wrap.type, value, config.mean, config.normal);
    }

    /**
     * 设置bitmap输入
     *
     * @param sourceBitmap bitmap
     * @param tensor       输入的Tensor
     * @param config       配置mean、normal、图片目标格式
     * @param matrix       定义缩放、旋转等
     * @return boolean
     */
    public static boolean convertBitmap(Bitmap sourceBitmap, MNNNetInstance.Session.Tensor tensor, Config config, Matrix matrix) {
        if (matrix == null) {
            matrix = new Matrix();
        }
        float[] value = new float[9];
        matrix.getValues(value);

        return MNNNetNative.nativeConvertBitmapToTensor(sourceBitmap, tensor.instance(),
                config.dest.type, config.filter.type, config.wrap.type, value, config.mean, config.normal);
    }
}
