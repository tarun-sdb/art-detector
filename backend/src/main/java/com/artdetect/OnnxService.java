package com.artdetect;

import ai.onnxruntime.*;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.FloatBuffer;
import java.util.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

/** Single-logit BCE model: input [1,3,S,S] float, output [1,1] logit. Sigmoid -> P(fake). */
@Service
public class OnnxService {
  private OrtEnvironment env;
  private OrtSession session;
  private final int size;
  private final float threshold;
  private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
  private static final float[] STD = {0.229f, 0.224f, 0.225f};

  public OnnxService(@Value("${model.path:models/model.onnx}") String path,
                     @Value("${model.size:384}") int size,
                     @Value("${model.threshold:0.5}") float threshold) throws OrtException {
    this.size = size; this.threshold = threshold;
    File f = new File(path);
    if (!f.exists()) return; // model missing -> 503 in controller, no fake predictions
    env = OrtEnvironment.getEnvironment();
    session = env.createSession(path, new OrtSession.SessionOptions());
  }

  public boolean ready() { return session != null; }

  public Map<String, Object> predict(byte[] bytes) throws Exception {
    BufferedImage img = ImageIO.read(new java.io.ByteArrayInputStream(bytes));
    if (img == null) throw new IllegalArgumentException("not an image");
    BufferedImage r = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
    Graphics2D g = r.createGraphics();
    g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
    g.drawImage(img, 0, 0, size, size, null); g.dispose();
    float[] data = new float[3 * size * size];
    for (int y = 0; y < size; y++) for (int x = 0; x < size; x++) {
      int rgb = r.getRGB(x, y);
      float[] c = {((rgb >> 16) & 255) / 255f, ((rgb >> 8) & 255) / 255f, (rgb & 255) / 255f};
      for (int k = 0; k < 3; k++) data[k * size * size + y * size + x] = (c[k] - MEAN[k]) / STD[k];
    }
    try (OnnxTensor t = OnnxTensor.createTensor(env, FloatBuffer.wrap(data), new long[]{1, 3, size, size});
         OrtSession.Result res = session.run(Map.of("input", t))) {
      float logit = ((float[][]) res.get(0).getValue())[0][0];
      double pFake = 1.0 / (1.0 + Math.exp(-logit));
      return Map.of("label", pFake >= threshold ? "ai" : "real",
          "p_fake", Math.round(pFake * 10000) / 10000.0, "threshold", threshold);
    }
  }
}
