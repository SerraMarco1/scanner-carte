package it.scannercarte.app;

import android.Manifest;
import android.content.Context;
import android.os.Build;
import android.os.VibrationEffect;
import android.os.Vibrator;
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
import androidx.camera.core.FocusMeteringAction;
import androidx.camera.core.FocusMeteringResult;
import androidx.camera.core.MeteringPoint;
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
import com.google.android.gms.tasks.Task;
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
import org.opencv.core.MatOfInt;
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
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayDeque;
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
    // Il testo si legge a doppia risoluzione (1280×1784): a 640×892 il numero in fondo alla carta spariva.
    // Le coordinate delle righe tornano alla pagina in scala 640×892, come nella versione web.
    private static final int OCR_SCALE = 2;
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
    // Ultimi fotogrammi della carta ridotti a celle 16×22: servono a vedere la lamina olografica (foilStats).
    private final ArrayDeque<float[]> foil = new ArrayDeque<>();
    private static final int GW = 16, GH = 22;
    private int still = 0, frameW = 0, frameH = 0;
    private long lastProcessed = 0, lastEvent = 0;
    private volatile String mode = "cerca";
    private volatile boolean needFocus = true;
    private volatile long focusStart = 0;      // > 0 mentre la fotocamera mette a fuoco sulla carta
    private static final long FOCUS_MS = 1500; // attesa massima della messa a fuoco
    private byte[] rowBuf;

    private OrtEnvironment env;
    private OrtSession session, edges;
    // Modello dei bordi (catalogo/bordi): ingresso RGB 192×256, uscita probabilità «carta» 96×128.
    private static final int EDGE_W = 192, EDGE_H = 256;
    private TextRecognizer latin, japanese, korean, chinese;
    private Bitmap lastCard;        // carta raddrizzata 1280×1784, per le letture asiatiche successive

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
                if (edges == null) {
                    // Sta nell'APK (pagina web, data/bordi.onnx); se manca resta la ricerca dei bordi con Canny.
                    try (InputStream in = getContext().getAssets().open("public/data/bordi.onnx")) {
                        ByteArrayOutputStream b = new ByteArrayOutputStream();
                        byte[] buf = new byte[1 << 16];
                        for (int n; (n = in.read(buf)) > 0; ) b.write(buf, 0, n);
                        OrtSession.SessionOptions eo = new OrtSession.SessionOptions();
                        eo.setIntraOpNumThreads(2);
                        edges = env.createSession(b.toByteArray(), eo);
                    } catch (Exception e) {
                        JSObject o = new JSObject();
                        o.put("message", "Modello dei bordi non caricato: " + e.getMessage());
                        notifyListeners("error", o);
                    }
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

    // Il modello (88 MB) non sta nell'APK: si scarica al primo avvio e resta nei file dell'app.
    // Stesso file e stessa versione usati per il catalogo e per data/probe.json.
    private static final String MODEL_URL = "https://huggingface.co/onnx-community/dinov2-small/resolve/8b1f705a3a7f6f062f6bdd21986c1583d3ef105d/onnx/model.onnx";
    private static final long MODEL_SIZE = 88532934L;

    private File modelFile() throws Exception {
        File f = new File(getContext().getFilesDir(), "model.onnx");
        if (f.length() == MODEL_SIZE) return f;
        File part = new File(getContext().getFilesDir(), "model.onnx.part");
        HttpURLConnection http = (HttpURLConnection) new URL(MODEL_URL).openConnection();
        http.setInstanceFollowRedirects(true);
        http.setConnectTimeout(20000);
        http.setReadTimeout(30000);
        try (InputStream in = http.getInputStream(); OutputStream out = new FileOutputStream(part)) {
            byte[] buf = new byte[1 << 16];
            long got = 0, told = 0;
            for (int n; (n = in.read(buf)) > 0; ) {
                out.write(buf, 0, n);
                got += n;
                if (got - told >= (1 << 20)) { // avanzamento per la barra della pagina, ogni MB
                    told = got;
                    JSObject o = new JSObject();
                    o.put("got", got);
                    o.put("total", MODEL_SIZE);
                    notifyListeners("download", o);
                }
            }
        } finally {
            http.disconnect();
        }
        if (part.length() != MODEL_SIZE) { part.delete(); throw new Exception("modello scaricato a metà: controlla la connessione e riapri l'app"); }
        if (!part.renameTo(f)) throw new Exception("non riesco a salvare il modello");
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
            if (now - lastProcessed < FRAME_MS || session == null || "risultato".equals(mode)) return;
            lastProcessed = now;
            Mat rgba = upright(toMat(image), image.getImageInfo().getRotationDegrees());
            Point[] q;
            try { q = edges != null ? findByModel(rgba) : findCard(rgba); }
            catch (Exception e) {
                // Un errore del modello non deve fermare lo scanner: si torna ai bordi con Canny e lo si segnala.
                edges = null;
                q = findCard(rgba);
                JSObject o = new JSObject();
                o.put("message", "Modello dei bordi: " + e);
                notifyListeners("error", o);
            }
            int w = rgba.cols(), h = rgba.rows();
            boolean movedNow = q != null && prevQuad != null && moved(q, prevQuad, Math.max(w, h));
            boolean focusing = focusStart > 0 && now - focusStart < FOCUS_MS;
            // Carta appena entrata, spostata o da rifare: prima la messa a fuoco su di lei, poi conta come ferma.
            // (Prova del 7/11: le scansioni sbagliate erano foto sfocate.)
            if (q != null && !focusing && (needFocus || prevQuad == null || movedNow)) { needFocus = false; focusOn(q, w, h); focusing = true; }
            still = q != null && prevQuad != null && !movedNow ? still + 1 : 0;
            prevQuad = q;
            float[] cells = q == null ? null : cellsOf(rgba, q);
            synchronized (lock) {
                if (lastFrame != null) lastFrame.release();
                lastFrame = rgba;
                lastQuad = q;
                frameW = w;
                frameH = h;
                if (cells == null) foil.clear();
                else { foil.addLast(cells); if (foil.size() > 10) foil.removeFirst(); }
            }
            final int s = focusing ? 0 : still; // durante la messa a fuoco i fotogrammi fermi si contano ma non si usano
            final Point[] found = q;
            overlay.post(() -> overlay.show(found, w, h, s, mode));
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

    // Carta raddrizzata e ridotta a celle GW×GH (media di ~10×10 pixel): RGB per cella.
    private static float[] cellsOf(Mat rgba, Point[] q) {
        Mat w = new Mat(), c = new Mat();
        MatOfPoint2f from = new MatOfPoint2f(q[0], q[1], q[2], q[3]);
        MatOfPoint2f to = new MatOfPoint2f(new Point(0, 0), new Point(GW * 10, 0), new Point(GW * 10, GH * 10), new Point(0, GH * 10));
        Mat m = Imgproc.getPerspectiveTransform(from, to);
        try {
            Imgproc.warpPerspective(rgba, w, m, new org.opencv.core.Size(GW * 10, GH * 10), Imgproc.INTER_LINEAR);
            Imgproc.resize(w, c, new org.opencv.core.Size(GW, GH), 0, 0, Imgproc.INTER_AREA);
            byte[] px = new byte[GW * GH * 4];
            c.get(0, 0, px);
            float[] f = new float[GW * GH * 3];
            for (int i = 0; i < GW * GH; i++) for (int k = 0; k < 3; k++) f[i * 3 + k] = px[i * 4 + k] & 0xFF;
            return f;
        } finally {
            w.release(); c.release(); m.release(); from.release(); to.release();
        }
    }

    // Lamina (reverse / holo): coi minimi movimenti del telefono i riflessi della lamina cambiano, la stampa no.
    // Per ogni cella la variazione del colore tra i fotogrammi (ognuno diviso per la sua luce media), in millesimi:
    // [finestra del disegno, resto della carta, fotogrammi]. Il bordo esterno si salta (risente dei contorni).
    // shortcut: solo misurata, non decide ancora la variante; da tarare con carte vere normali, reverse e holo.
    private double[] foilStats() {
        List<float[]> fr;
        synchronized (lock) { fr = new ArrayList<>(foil); }
        int n = fr.size();
        if (n < 3) return null;
        double[][] norm = new double[n][];
        for (int t = 0; t < n; t++) {
            float[] f = fr.get(t);
            double[] mean = new double[3];
            for (int i = 0; i < f.length; i++) mean[i % 3] += f[i];
            norm[t] = new double[f.length];
            for (int i = 0; i < f.length; i++) norm[t][i] = f[i] / Math.max(1, mean[i % 3] / (GW * GH));
        }
        double art = 0, body = 0;
        int na = 0, nb = 0;
        for (int y = 1; y < GH - 1; y++)
            for (int x = 1; x < GW - 1; x++) {
                boolean inArt = x >= 2 && x <= 13 && y >= 3 && y <= 10, outArt = y <= 1 || y >= 12; // celle tutte dentro o tutte fuori
                if (!inArt && !outArt) continue;
                double v = 0;
                for (int k = 0; k < 3; k++) {
                    int i = (y * GW + x) * 3 + k;
                    double s = 0, s2 = 0;
                    for (int t = 0; t < n; t++) { s += norm[t][i]; s2 += norm[t][i] * norm[t][i]; }
                    v += Math.sqrt(Math.max(0, s2 / n - (s / n) * (s / n)));
                }
                if (inArt) { art += v; na++; } else { body += v; nb++; }
            }
        return new double[] { 1000 * art / na, 1000 * body / nb, n };
    }

    // Messa a fuoco ed esposizione sul centro della carta (q in pixel del fotogramma fw×fh, mappati come in QuadView).
    private void focusOn(Point[] q, int fw, int fh) {
        focusStart = SystemClock.uptimeMillis();
        double cx = (q[0].x + q[1].x + q[2].x + q[3].x) / 4, cy = (q[0].y + q[1].y + q[2].y + q[3].y) / 4;
        previewView.post(() -> {
            try {
                if (camera == null) { focusStart = 0; return; }
                float vw = previewView.getWidth(), vh = previewView.getHeight(), s = Math.max(vw / fw, vh / fh);
                MeteringPoint p = previewView.getMeteringPointFactory().createPoint((vw - fw * s) / 2 + (float) cx * s, (vh - fh * s) / 2 + (float) cy * s);
                ListenableFuture<FocusMeteringResult> f = camera.getCameraControl().startFocusAndMetering(
                    new FocusMeteringAction.Builder(p, FocusMeteringAction.FLAG_AF | FocusMeteringAction.FLAG_AE).setAutoCancelDuration(3, TimeUnit.SECONDS).build());
                f.addListener(() -> focusStart = 0, ContextCompat.getMainExecutor(getContext()));
            } catch (Exception e) {
                focusStart = 0;
            }
        });
    }

    // Come findByModel di web/bordi.js: il modello vede il centro 3:4 verticale del fotogramma.
    private Point[] findByModel(Mat rgba) throws Exception {
        int w = rgba.cols(), h = rgba.rows();
        int cw = (int) Math.min(w, h * 0.75), ch = (int) Math.min(h, Math.round(cw / 0.75)), cx = (w - cw) / 2, cy = (h - ch) / 2;
        Mat crop = new Mat(rgba, new org.opencv.core.Rect(cx, cy, cw, ch)), small = new Mat();
        try {
            Imgproc.resize(crop, small, new org.opencv.core.Size(EDGE_W, EDGE_H), 0, 0, Imgproc.INTER_AREA);
            int n = EDGE_W * EDGE_H;
            byte[] px = new byte[n * 4];
            small.get(0, 0, px);
            float[] t = new float[3 * n];
            for (int i = 0; i < n; i++) for (int c = 0; c < 3; c++) t[c * n + i] = px[i * 4 + c] & 0xFF;
            float[] prob = new float[n / 4];
            try (OnnxTensor x = OnnxTensor.createTensor(env, FloatBuffer.wrap(t), new long[] { 1, 3, EDGE_H, EDGE_W });
                 OrtSession.Result r = edges.run(Collections.singletonMap("image", x))) {
                ((OnnxTensor) r.get(0)).getFloatBuffer().get(prob);
            }
            Point[] q = quadFromMask(prob);
            if (q == null) return null;
            for (Point p : q) { p.x = cx + p.x * cw / EDGE_W; p.y = cy + p.y * ch / EDGE_H; }
            return q;
        } finally {
            crop.release(); small.release();
        }
    }

    // Maschera 96×128 → angoli nell'ingresso 192×256: contorno più grande, involucro convesso, poligono a 4 lati
    // che lo segue meglio (o rettangolo ruotato che lo contiene). Come quadFromMask di web/bordi.js.
    private static Point[] quadFromMask(float[] prob) {
        Mat p = new Mat(EDGE_H / 2, EDGE_W / 2, CvType.CV_32F), big = new Mat(), m = new Mat(), hier = new Mat();
        List<MatOfPoint> cs = new ArrayList<>();
        try {
            p.put(0, 0, prob);
            Imgproc.resize(p, big, new org.opencv.core.Size(EDGE_W, EDGE_H), 0, 0, Imgproc.INTER_LINEAR);
            Imgproc.threshold(big, big, 0.5, 255, Imgproc.THRESH_BINARY);
            big.convertTo(m, CvType.CV_8U);
            Imgproc.findContours(m, cs, hier, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
            MatOfPoint best = null;
            double bestArea = 0.06 * EDGE_W * EDGE_H;
            for (MatOfPoint c : cs) { double a = Imgproc.contourArea(c); if (a > bestArea) { best = c; bestArea = a; } }
            if (best == null) return null;
            MatOfInt idx = new MatOfInt();
            Imgproc.convexHull(best, idx);
            Point[] pts = best.toArray();
            int[] ids = idx.toArray();
            idx.release();
            Point[] hp = new Point[ids.length];
            for (int k = 0; k < ids.length; k++) hp[k] = pts[ids[k]];
            MatOfPoint2f hull = new MatOfPoint2f(hp), poly = new MatOfPoint2f();
            double peri = Imgproc.arcLength(hull, true);
            Point[] q = null;
            for (double eps : new double[] { 0.02, 0.03, 0.04, 0.05, 0.06, 0.08 }) {
                Imgproc.approxPolyDP(hull, poly, eps * peri, true);
                if (poly.total() == 4) { q = order(poly.toArray()); break; }
                if (poly.total() < 4) break;
            }
            if (q == null) { Point[] b = new Point[4]; Imgproc.minAreaRect(hull).points(b); q = order(b); }
            hull.release(); poly.release();
            q = straighten(q, pts);
            return cardShaped(q) ? q : null;
        } finally {
            p.release(); big.release(); m.release(); hier.release();
            for (MatOfPoint c : cs) c.release();
        }
    }

    // Angoli più precisi: per ogni lato una retta sui punti del contorno lontani dagli angoli (arrotondati) e dalle dita,
    // poi gli incroci delle rette. Come straighten in web/bordi.js.
    private static Point[] straighten(Point[] q, Point[] pts) {
        double[][] lines = new double[4][];
        for (int k = 0; k < 4; k++) {
            Point a = q[k], b = q[(k + 1) % 4];
            double dx = b.x - a.x, dy = b.y - a.y, n = Math.hypot(dx, dy), ux = dx / n, uy = dy / n;
            List<Point> sel = new ArrayList<>();
            for (Point p : pts) {
                double t = (p.x - a.x) * ux + (p.y - a.y) * uy, off = Math.abs((p.y - a.y) * ux - (p.x - a.x) * uy);
                if (t > 0.15 * n && t < 0.85 * n && off < 0.04 * n + 2) sel.add(p);
            }
            if (sel.size() < 5) return q;
            MatOfPoint2f m = new MatOfPoint2f(sel.toArray(new Point[0]));
            Mat l = new Mat();
            Imgproc.fitLine(m, l, Imgproc.DIST_HUBER, 0, 0.01, 0.01);
            lines[k] = new double[] { l.get(0, 0)[0], l.get(1, 0)[0], l.get(2, 0)[0], l.get(3, 0)[0] }; // vx, vy, x0, y0
            m.release(); l.release();
        }
        Point[] out = new Point[4];
        for (int k = 0; k < 4; k++) {
            double[] A = lines[(k + 3) % 4], B = lines[k];
            double det = B[0] * A[1] - A[0] * B[1];
            if (Math.abs(det) < 1e-6) return q;
            double s = (B[0] * (B[3] - A[3]) - B[1] * (B[2] - A[2])) / det; // incrocio: p + s·a = r + t·b
            out[k] = new Point(A[2] + s * A[0], A[3] + s * A[1]);
            if (Math.max(Math.abs(out[k].x - q[k].x), Math.abs(out[k].y - q[k].y)) >= 0.1 * EDGE_W) return q;
        }
        return out;
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
                // La lettura del testo (ML Kit, thread suoi) parte subito e lavora mentre si calcolano le impronte.
                Bitmap card = Bitmap.createBitmap(OCR_W * OCR_SCALE, OCR_H * OCR_SCALE, Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(hi, card);
                lastCard = card;
                Task<Text> ocr = latinReader().process(InputImage.fromBitmap(card, 0));
                Imgproc.resize(hi, small, new org.opencv.core.Size(CARD_W, CARD_H), 0, 0, Imgproc.INTER_AREA);
                // Finestra dell'illustrazione: stessi arrotondamenti di artRect() in web/core.js.
                org.opencv.core.Rect art = new org.opencv.core.Rect((int) Math.round(ART_X * CARD_W), (int) Math.round(ART_Y * CARD_H),
                    (int) Math.round(ART_W * CARD_W), (int) Math.round(ART_H * CARD_H));
                Mat artMat = new Mat(small, art);
                float[] va = embed(artMat), vf = embed(small);
                artMat.release();
                long t1 = SystemClock.uptimeMillis();
                JSArray lines = lines(ocr);
                JSObject o = new JSObject();
                double[] f = foilStats();
                if (f != null) { JSArray a = new JSArray(); for (double x : f) a.put(Double.valueOf(Math.round(x * 10) / 10.0)); o.put("foil", a); }
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
        MatOfPoint2f to = new MatOfPoint2f(new Point(0, 0), new Point(OCR_W * OCR_SCALE, 0), new Point(OCR_W * OCR_SCALE, OCR_H * OCR_SCALE), new Point(0, OCR_H * OCR_SCALE));
        Mat m = Imgproc.getPerspectiveTransform(from, to);
        Imgproc.warpPerspective(frame, out, m, new org.opencv.core.Size(OCR_W * OCR_SCALE, OCR_H * OCR_SCALE), Imgproc.INTER_LINEAR);
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
        return lines(reader.process(InputImage.fromBitmap(card, 0)));
    }

    private static JSArray lines(Task<Text> task) throws Exception {
        Text text = Tasks.await(task, 10, TimeUnit.SECONDS);
        JSArray out = new JSArray();
        for (Text.TextBlock block : text.getTextBlocks())
            for (Text.Line line : block.getLines()) {
                android.graphics.Rect b = line.getBoundingBox();
                if (b == null) continue;
                JSObject o = new JSObject();
                o.put("text", line.getText());
                o.put("x", b.left / OCR_SCALE);
                o.put("y", b.top / OCR_SCALE);
                o.put("w", b.width() / OCR_SCALE);
                o.put("h", b.height() / OCR_SCALE);
                out.put(o);
            }
        return out;
    }

    // ---------- comandi ----------

    @PluginMethod
    public void setMode(PluginCall call) {
        mode = call.getString("mode", "cerca");
        if ("cerca".equals(mode)) needFocus = true; // carta nuova o scansione da rifare: rimettere a fuoco
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

    // Vibrazione del semaforo: pattern in ms (vibra, pausa, vibra…). Nella WebView navigator.vibrate non funziona.
    @PluginMethod
    public void vibrate(PluginCall call) {
        try {
            JSArray p = call.getArray("pattern", new JSArray());
            long[] ms = new long[p.length() + 1]; // il primo valore di Android è l'attesa iniziale
            for (int i = 0; i < p.length(); i++) ms[i + 1] = p.getLong(i);
            Vibrator v = (Vibrator) getContext().getSystemService(Context.VIBRATOR_SERVICE);
            if (v != null && v.hasVibrator() && p.length() > 0) {
                if (Build.VERSION.SDK_INT >= 26) v.vibrate(VibrationEffect.createWaveform(ms, -1));
                else v.vibrate(ms, -1);
            }
            call.resolve();
        } catch (Exception e) {
            call.reject("Vibrazione: " + e.getMessage());
        }
    }

    @PluginMethod
    public void info(PluginCall call) {
        call.resolve(cameraInfo());
    }

    // Errore che ha chiuso l'app l'ultima volta (scritto da MainActivity), poi cancellato.
    @PluginMethod
    public void lastCrash(PluginCall call) {
        File f = new File(getContext().getFilesDir(), "crash.txt");
        JSObject o = new JSObject();
        try {
            if (f.exists()) {
                o.put("text", new String(java.nio.file.Files.readAllBytes(f.toPath())));
                f.delete();
            }
        } catch (Exception e) {
            o.put("text", "crash.txt illeggibile: " + e.getMessage());
        }
        call.resolve(o);
    }

    // ---------- utilità ----------

    private static JSArray toJs(float[] v) {
        JSArray a = new JSArray();
        for (float f : v) a.put(Double.valueOf(f)); // put(Object): put(double) dichiara JSONException
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
