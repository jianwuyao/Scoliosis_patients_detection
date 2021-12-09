package org.jwyao.mnnpose.mnn;

public enum MNNForwardType {
    FORWARD_CPU(0),
    FORWARD_OPENCL(3),
    FORWARD_AUTO(4),
    FORWARD_OPENGL(6),
    FORWARD_VULKAN(7);  // 后两个板子不支持

    public int type;

    MNNForwardType(int t) {
        type = t;
    }
}
