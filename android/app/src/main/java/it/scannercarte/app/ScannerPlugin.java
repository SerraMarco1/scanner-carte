package it.scannercarte.app;

import android.Manifest;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.SystemClock;
import android.util.Base64;
import android.util.Size;
import android.view.ViewGroup;
import android.webkit.WebView;

import androidx.camera.core.Camera;
import androidx.camera.core.CameraInfo;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.resolutionselector.AspectRatioStrategy;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.getcapacitor.JSArray;
import com.getcapacitor.JSObject;
import com.getcapacitor.PermissionState;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import com.getcapacitor.annotation.Permission;
import com.getcapacitor.annotation.PermissionCallback;
import com.google.android.gms.tasks.Tasks;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.Text;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions;
import com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions;
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

/**
 * La parte pesante dello scanner, nativa: fotocamera (CameraX) dietro la pagina, ricerca dei bordi
 * della carta (OpenCV), impronte del disegno (ONNX Runtime) e lettura del testo (ML Kit).
 * La pagina (web/app.js) tiene interfaccia e logica, uguali alla versione web per iPhone.
 * Gli stessi ritagli e le stesse formule di web/core.js e web/bordi.js: se cambiano là, cambiano qui.
 */
@CapacitorPlugin(name = "Scanner", permissions = { @Permission(strings = { Manifest.permission.CAMERA }, alias = "camera") })
public class ScannerPlugin extends Plugin {
    // Come web/core.js
    private static final int CARD_W = 320, CARD_H = 446, OCR_W = 640, OCR_H = 892, SIZE = 224, DIM = 384;
    private static final double ART_X = 0.08, ART_Y = 0.10, ART_W = 0.84, ART_H = 0.44;
    private static final float[] MEAN = { 0.485f, 0.456f, 0.406f }, STD = { 0.229f, 0.224f, 0.225f };
    // Come web/bordi.js
    private static final int DET = 480;
    private static final double MOVE = 0.025; // spostamento tollerato tra due fotogrammi, in frazione del lato lungo
    private static final long FRAME_MS = 80;  // al massimo ~12 fotogrammi analizzati al secondo

    private PreviewView previewView;
    private QuadView overlay;
    private ProcessCameraProvider provider;
    private Camera camera;
    private List<CameraInfo> backCameras = new ArrayList<>();
    private int camIndex = 0;
    private final ExecutorService frames = Executors.newSingleThreadExecutor();
    private final ExecutorService work = Executors.newSingleThreadExecutor();
    private final Object lock = new Object();
    private Mat lastFrame;          // ultimo fotogramma, RGBA e già girato in verticale
    private Point[] lastQuad, prevQuad;
    private int still = 0, frameW = 0, frameH = 0;
    private long lastProcessed = 0, lastEvent = 0;
    private volatile String mode = "cerca";
    private byte[] rowBuf;

    private OrtEnvironment env;
    private OrtSession session;
    private TextRecognizer latin, japanese, korean, chinese;
    private Bitmap lastCard;        // carta raddrizzata 640×892, per le letture asiatiche successive

    // ---------- avvio ----------

    @PluginMethod
    public void start(PluginCall call) {
        if (getPermissionState("camera") != PermissionState.GRANTED) {
            requestPermissionForAlias("camera", call, "cameraPermission");
            return;
        }
        startCamera(call);
    }

    @PermissionCallback
    private void cameraPermission(PluginCall call) {
        if (getPermissionState("camera") == PermissionState.GRANTED) startCamera(call);
        else call.reject("Permesso della fotocamera negato");
    }

    private void startCamera(PluginCall call) {
        getActivity().runOnUiThread(() -> {
            try {
                if (previewView == null) {
                    // Fotocamera e contorno dietro la pagina, che diventa trasparente.
                    WebView web = getBridge().getWebView();
                    web.setBackgroundColor(Color.TRANSPARENT);
                    ViewGroup parent = (ViewGroup) web.getParent();
                    previewView = new PreviewView(getContext());
                    previewView.setScaleType(PreviewView.ScaleType.FILL_CENTER);
                    previewView.setImplementationMode(PreviewView.ImplementationMode.COMPATIBLE);
                    overlay = new QuadView(getContext());
                    parent.addView(previewView, 0, new ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
                    parent.addView(overlay, 1, new ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
                }
                ListenableFuture<ProcessCameraProvider> future = ProcessCameraProvider.getInstance(getContext());
                future.addListener(() -> {
                    try {
                        provider = future.get();
                        backCameras = new ArrayList<>();
                        for (CameraInfo info : provider.getAvailableCameraInfos())
                            if (info.getLensFacing() == CameraSelector.LENS_FACING_BACK) backCameras.add(info);
                        if (backCameras.isEmpty()) { call.reject("Nessuna fotocamera posteriore"); return; }
                        bindCamera();
                        call.resolve(cameraInfo());
                    } catch (Exception e) {
                        call.reject("Fotocamera: " + e.getMessage());
                    }
                }, ContextCompat.getMainExecutor(getContext()));
            } catch (Exception e) {
                call.reject("Fotocamera: " + e.getMessage());
            }
        });
    }

    private void bindCamera() {
        provider.unbindAll();
        // 4:3 = tutto il sensore (il 16:9 del video del browser ne taglia una parte).
        ResolutionSelector analysisRes = new ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .setResolutionStrategy(new ResolutionStrategy(new Size(1920, 1440), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
            .build();
        ResolutionSelector previewRes = new ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY).build();
        Preview preview = new Preview.Builder().setResolutionSelector(previewRes).build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        ImageAnalysis analysis = new ImageAnalysis.Builder()
            .setResolutionSelector(analysisRes)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build();
        analysis.setAnalyzer(frames, this::onFrame);
        CameraSelector selector = backCameras.get(camIndex % backCameras.size()).getCameraSelector();
        camera = provider.bindToLifecycle((LifecycleOwner) getActivity(), selector, preview, analysis);
    }

    private JSObject cameraInfo() {
        JSObject o = new JSObject();
        o.put("cameras", backCameras.size());
        o.put("index", camIndex % Math.max(1, backCameras.size()));
        o.put("frame", frameW + "x" + frameH);
        o.put("torch", camera != null && camera.getCameraInfo().hasFlashUnit());
        return o;
    }

    // Modello delle impronte e verifica: la pagina confronta il risultato con data/probe.json.
    @PluginMethod
    public void prepare(PluginCall call) {
        work.execute(() -> {
            try {
                long t0 = SystemClock.uptimeMillis();
                if (!OpenCVLoader.initLocal()) { call.reject("OpenCV non si avvia"); return; }
                if (session == null) {
                    env = OrtEnvironment.getEnvironment();
                    OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
                    opts.setIntraOpNumThreads(4);
                    session = env.createSession(modelFile().getAbsolutePath(), opts);
                }
                float[] probe = run(probeInput());
                JSObject o = new JSObject();
                o.put("probe", toJs(probe));
                o.put("ms", SystemClock.uptimeMillis() - t0);
                call.resolve(o);
            } catch (Throwable e) {
                call.reject("Modello: " + e.getMessage());
            }
        });
    }

    // ONNX Runtime legge il modello da file: la prima volta lo si copia dall'APK.
    private File modelFile() throws Exception {
        File f = new File(getContext().getFilesDir(), "model.onnx");
        long size;
        try (InputStream in = getContext().getAssets().open("model.onnx")) { size = in.available(); }
        if (f.exists() && f.length() > 0 && f.length() >= size) return f;
        try (InputStream in = getContext().getAssets().open("model.onnx"); OutputStream out = new FileOutputStream(f)) {
            byte[] buf = new byte[1 << 20];
            for (int n; (n = in.read(buf)) > 0; ) out.write(buf, 0, n);
        }
        return f;
    }

    // Come probeInput() di web/core.js.
    private static float[] probeInput() {
        float[] out = new float[3 * SIZE * SIZE];
        for (int i = 0; i < out.length; i++) out[i] = (float) (1.5 * Math.sin(i * 0.0011) * Math.cos(i * 0.00037 + (i % SIZE) * 0.05));
        return out;
    }

    // ---------- fotogrammi: bordi della carta ----------

    private void onFrame(ImageProxy image) {
        try {
            long now = SystemClock.uptimeMillis();
            if (now - lastProcessed < FRAME_MS || session == null) return;
            lastProcessed = now;
            Mat rgba = upright(toMat(image), image.getImageInfo().getRotationDegrees());
            Point[] q = findCard(rgba);
            int w = rgba.cols(), h = rgba.rows();
            still = q != null && prevQuad != null && !moved(q, prevQuad, Math.max(w, h)) ? still + 1 : 0;
            prevQuad = q;
            synchronized (lock) {
                if (lastFrame != null) lastFrame.release();
                lastFrame = rgba;
                lastQuad = q;
                frameW = w;
                frameH = h;
            }
            final int s = still;
            overlay.post(() -> overlay.show(q, w, h, s, mode));
            if (now - lastEvent >= 90) {
                lastEvent = now;
                JSObject o = new JSObject();
                o.put("found", q != null);
                o.put("still", s);
                notifyListeners("frame", o);
            }
        } catch (Throwable e) {
            JSObject o = new JSObject();
            o.put("message", "Fotogramma: " + e.getMessage());
            notifyListeners("error", o);
        } finally {
            image.close();
        }
    }

    private Mat toMat(ImageProxy image) {
        ImageProxy.PlaneProxy plane = image.getPlanes()[0];
        ByteBuffer buf = plane.getBuffer();
        int w = image.getWidth(), h = image.getHeight(), stride = plane.getRowStride();
        Mat m = new Mat(h, w, CvType.CV_8UC4);
        if (rowBuf == null || rowBuf.length != w * 4) rowBuf = new byte[w * 4];
        for (int y = 0; y < h; y++) {
            buf.position(y * stride);
            buf.get(rowBuf, 0, w * 4);
            m.put(y, 0, rowBuf);
        }
        return m;
    }

    private static Mat upright(Mat m, int degrees) {
        if (degrees == 0) return m;
        Mat r = new Mat();
        Core.rotate(m, r, degrees == 90 ? Core.ROTATE_90_CLOCKWISE : degrees == 180 ? Core.ROTATE_180 : Core.ROTATE_90_COUNTERCLOCKWISE);
        m.release();
        return r;
    }

    // Come findCard di web/bordi.js: tre sensibilità dei bordi, tutti i contorni, 4 lati o rettangolo che li contiene.
    private static Point[] findCard(Mat rgba) {
        double k = DET / (double) Math.max(rgba.cols(), rgba.rows());
        Mat small = new Mat(), gray = new Mat(), edges = new Mat(), hier = new Mat(), kernel = Mat.ones(3, 3, CvType.CV_8U);
        try {
            Imgproc.resize(rgba, small, new org.opencv.core.Size(Math.round(rgba.cols() * k), Math.round(rgba.rows() * k)), 0, 0, Imgproc.INTER_AREA);
            Imgproc.cvtColor(small, gray, Imgproc.COLOR_RGBA2GRAY);
            Imgproc.GaussianBlur(gray, gray, new org.opencv.core.Size(5, 5), 0);
            Point[] best = null;
            double bestArea = small.cols() * small.rows() * 0.06;
            int[][] thresholds = { { 20, 60 }, { 40, 120 }, { 80, 200 } };
            for (int[] t : thresholds) {
                Imgproc.Canny(gray, edges, t[0], t[1]);
                Imgproc.dilate(edges, edges, kernel);
                List<MatOfPoint> contours = new ArrayList<>();
                Imgproc.findContours(edges, contours, hier, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
                for (MatOfPoint c : contours) {
                    double area = Imgproc.contourArea(c);
                    if (area > bestArea) {
                        MatOfPoint2f c2 = new MatOfPoint2f(c.toArray()), poly = new MatOfPoint2f();
                        Imgproc.approxPolyDP(c2, poly, 0.03 * Imgproc.arcLength(c2, true), true);
                        Point[] q = null;
                        if (poly.total() == 4) {
                            MatOfPoint p = new MatOfPoint(poly.toArray());
                            if (Imgproc.isContourConvex(p)) q = order(poly.toArray());
                            p.release();
                        }
                        if (q == null) {
                            RotatedRect r = Imgproc.minAreaRect(c2);
                            if (area > 0.85 * r.size.area()) { Point[] pts = new Point[4]; r.points(pts); q = order(pts); }
                        }
                        if (q != null && cardShaped(q)) { best = q; bestArea = area; }
                        c2.release();
                        poly.release();
                    }
                    c.release();
                }
            }
            if (best == null) return null;
            for (Point p : best) { p.x /= k; p.y /= k; }
            return best;
        } finally {
            small.release(); gray.release(); edges.release(); hier.release(); kernel.release();
        }
    }

    // Angoli in ordine: alto-sinistra, alto-destra, basso-destra, basso-sinistra.
    private static Point[] order(Point[] p) {
        Point tl = p[0], tr = p[0], br = p[0], bl = p[0];
        for (Point q : p) {
            if (q.x + q.y < tl.x + tl.y) tl = q;
            if (q.x + q.y > br.x + br.y) br = q;
            if (q.y - q.x < tr.y - tr.x) tr = q;
            if (q.y - q.x > bl.y - bl.x) bl = q;
        }
        return new Point[] { new Point(tl.x, tl.y), new Point(tr.x, tr.y), new Point(br.x, br.y), new Point(bl.x, bl.y) };
    }

    private static double dist(Point a, Point b) { return Math.hypot(a.x - b.x, a.y - b.y); }

    // Carta in verticale: larghezza/altezza ≈ 63/88 = 0,72, con margine per la prospettiva.
    private static boolean cardShaped(Point[] q) {
        double r = (dist(q[0], q[1]) + dist(q[3], q[2])) / (dist(q[0], q[3]) + dist(q[1], q[2]));
        return r > 0.55 && r < 0.9;
    }

    private static boolean moved(Point[] a, Point[] b, int size) {
        for (int i = 0; i < 4; i++) if (dist(a[i], b[i]) > MOVE * size) return true;
        return false;
    }

    // ---------- analisi della carta ferma ----------

    @PluginMethod
    public void analyze(PluginCall call) {
        work.execute(() -> {
            Mat frame;
            Point[] q;
            synchronized (lock) {
                frame = lastFrame == null ? null : lastFrame.clone();
                q = lastQuad;
            }
            if (frame == null || q == null) { call.reject("Nessuna carta inquadrata"); return; }
            Mat hi = new Mat(), small = new Mat();
            try {
                long t0 = SystemClock.uptimeMillis();
                warp(frame, q, hi);
                Imgproc.resize(hi, small, new org.opencv.core.Size(CARD_W, CARD_H), 0, 0, Imgproc.INTER_AREA);
                // Finestra dell'illustrazione: stessi arrotondamenti di artRect() in web/core.js.
                org.opencv.core.Rect art = new org.opencv.core.Rect((int) Math.round(ART_X * CARD_W), (int) Math.round(ART_Y * CARD_H),
                    (int) Math.round(ART_W * CARD_W), (int) Math.round(ART_H * CARD_H));
                Mat artMat = new Mat(small, art);
                float[] va = embed(artMat), vf = embed(small);
                artMat.release();
                long t1 = SystemClock.uptimeMillis();
                Bitmap card = Bitmap.createBitmap(OCR_W, OCR_H, Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(hi, card);
                lastCard = card;
                JSArray lines = read(latinReader(), card);
                JSObject o = new JSObject();
                o.put("art", toJs(va));
                o.put("full", toJs(vf));
                o.put("lines", lines);
                o.put("seen", jpeg(small));
                o.put("msEmbed", t1 - t0);
                o.put("msOcr", SystemClock.uptimeMillis() - t1);
                o.put("frame", frame.cols() + "x" + frame.rows());
                call.resolve(o);
            } catch (Throwable e) {
                call.reject("Analisi: " + e.getMessage());
            } finally {
                frame.release(); hi.release(); small.release();
            }
        });
    }

    private static void warp(Mat frame, Point[] q, Mat out) {
        MatOfPoint2f from = new MatOfPoint2f(q[0], q[1], q[2], q[3]);
        MatOfPoint2f to = new MatOfPoint2f(new Point(0, 0), new Point(OCR_W, 0), new Point(OCR_W, OCR_H), new Point(0, OCR_H));
        Mat m = Imgproc.getPerspectiveTransform(from, to);
        Imgproc.warpPerspective(frame, out, m, new org.opencv.core.Size(OCR_W, OCR_H), Imgproc.INTER_LINEAR);
        from.release(); to.release(); m.release();
    }

    // Come toTensorData + normalize di web/core.js: RGB, 224×224, normalizzato, token CLS.
    private float[] embed(Mat rgba) throws Exception {
        Mat r = new Mat();
        Imgproc.resize(rgba, r, new org.opencv.core.Size(SIZE, SIZE), 0, 0, Imgproc.INTER_AREA);
        byte[] px = new byte[SIZE * SIZE * 4];
        r.get(0, 0, px);
        r.release();
        int n = SIZE * SIZE;
        float[] t = new float[3 * n];
        for (int i = 0; i < n; i++)
            for (int c = 0; c < 3; c++) t[c * n + i] = ((px[i * 4 + c] & 0xFF) / 255f - MEAN[c]) / STD[c];
        return run(t);
    }

    private float[] run(float[] input) throws Exception {
        try (OnnxTensor x = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), new long[] { 1, 3, SIZE, SIZE });
             OrtSession.Result res = session.run(Collections.singletonMap("pixel_values", x))) {
            FloatBuffer fb = ((OnnxTensor) res.get(0)).getFloatBuffer();
            float[] v = new float[DIM];
            fb.get(v, 0, DIM);
            double s = 0;
            for (float f : v) s += f * f;
            s = Math.sqrt(s);
            if (s > 0) for (int i = 0; i < DIM; i++) v[i] /= (float) s;
            return v;
        }
    }

    // ---------- lettura del testo (ML Kit) ----------

    private TextRecognizer latinReader() {
        if (latin == null) latin = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);
        return latin;
    }

    // Seconda lettura sulla stessa carta, quando il lettore latino non ha trovato la lingua.
    @PluginMethod
    public void readAsian(PluginCall call) {
        String script = call.getString("script", "japanese");
        work.execute(() -> {
            try {
                if (lastCard == null) { call.reject("Nessuna carta letta"); return; }
                TextRecognizer r;
                if ("korean".equals(script)) { if (korean == null) korean = TextRecognition.getClient(new KoreanTextRecognizerOptions.Builder().build()); r = korean; }
                else if ("chinese".equals(script)) { if (chinese == null) chinese = TextRecognition.getClient(new ChineseTextRecognizerOptions.Builder().build()); r = chinese; }
                else { if (japanese == null) japanese = TextRecognition.getClient(new JapaneseTextRecognizerOptions.Builder().build()); r = japanese; }
                JSObject o = new JSObject();
                o.put("lines", read(r, lastCard));
                call.resolve(o);
            } catch (Throwable e) {
                call.reject("Lettura: " + e.getMessage());
            }
        });
    }

    private static JSArray read(TextRecognizer reader, Bitmap card) throws Exception {
        Text text = Tasks.await(reader.process(InputImage.fromBitmap(card, 0)), 10, TimeUnit.SECONDS);
        JSArray out = new JSArray();
        for (Text.TextBlock block : text.getTextBlocks())
            for (Text.Line line : block.getLines()) {
                android.graphics.Rect b = line.getBoundingBox();
                if (b == null) continue;
                JSObject o = new JSObject();
                o.put("text", line.getText());
                o.put("x", b.left);
                o.put("y", b.top);
                o.put("w", b.width());
                o.put("h", b.height());
                out.put(o);
            }
        return out;
    }

    // ---------- comandi ----------

    @PluginMethod
    public void setMode(PluginCall call) {
        mode = call.getString("mode", "cerca");
        call.resolve();
    }

    @PluginMethod
    public void torch(PluginCall call) {
        boolean on = Boolean.TRUE.equals(call.getBoolean("on", false));
        if (camera == null) { call.reject("Fotocamera spenta"); return; }
        camera.getCameraControl().enableTorch(on);
        call.resolve();
    }

    @PluginMethod
    public void switchCamera(PluginCall call) {
        getActivity().runOnUiThread(() -> {
            try {
                camIndex = (camIndex + 1) % Math.max(1, backCameras.size());
                bindCamera();
                call.resolve(cameraInfo());
            } catch (Exception e) {
                call.reject("Obiettivo: " + e.getMessage());
            }
        });
    }

    @PluginMethod
    public void info(PluginCall call) {
        call.resolve(cameraInfo());
    }

    // ---------- utilità ----------

    private static JSArray toJs(float[] v) {
        JSArray a = new JSArray();
        for (float f : v) a.put((double) f);
        return a;
    }

    private static String jpeg(Mat rgba) {
        Bitmap b = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rgba, b);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        b.compress(Bitmap.CompressFormat.JPEG, 85, out);
        return "data:image/jpeg;base64," + Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP);
    }
}
