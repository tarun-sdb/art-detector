package com.artdetect;

import java.util.Map;
import org.springframework.http.*;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@Controller
public class PredictController {
  private final OnnxService onnx;
  public PredictController(OnnxService onnx) { this.onnx = onnx; }

  @GetMapping("/") public String index() { return "index"; }

  @PostMapping(value = "/api/predict", produces = MediaType.APPLICATION_JSON_VALUE)
  @ResponseBody
  public ResponseEntity<?> predict(@RequestParam("file") MultipartFile file) {
    try {
      if (!onnx.ready()) return ResponseEntity.status(503)
          .body(Map.of("error", "model not loaded: train first (ml/train.py) and set model.path"));
      if (file.isEmpty() || file.getSize() > 10 * 1024 * 1024)
        return ResponseEntity.badRequest().body(Map.of("error", "empty or >10MB"));
      return ResponseEntity.ok(onnx.predict(file.getBytes()));
    } catch (IllegalArgumentException e) {
      return ResponseEntity.badRequest().body(Map.of("error", e.getMessage()));
    } catch (Exception e) {
      return ResponseEntity.status(500).body(Map.of("error", "inference failed"));
    }
  }
}
