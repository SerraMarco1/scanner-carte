package it.scannercarte.app;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.view.View;

import org.opencv.core.Point;

/** Contorno della carta trovata, disegnato sopra l'anteprima della fotocamera (che riempie lo schermo tagliando i bordi). */
public class QuadView extends View {
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Path path = new Path();
    private float[] pts;

    public QuadView(Context context) {
        super(context);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(4 * context.getResources().getDisplayMetrics().density);
        setWillNotDraw(false);
    }

    /** q in pixel del fotogramma (fw×fh); colori come nella versione web: giallo trovata, verde ferma, grigio «toglila». */
    public void show(Point[] q, int fw, int fh, int still, String mode) {
        if (q == null || fw == 0 || fh == 0) {
            pts = null;
        } else {
            float vw = getWidth(), vh = getHeight(), s = Math.max(vw / fw, vh / fh);
            float ox = (vw - fw * s) / 2, oy = (vh - fh * s) / 2;
            pts = new float[8];
            for (int i = 0; i < 4; i++) { pts[2 * i] = ox + (float) q[i].x * s; pts[2 * i + 1] = oy + (float) q[i].y * s; }
        }
        paint.setColor("via".equals(mode) ? 0xFF9AA1AE : still > 0 ? 0xFF3CCF7D : 0xFFFFD23F);
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        if (pts == null) return;
        path.reset();
        path.moveTo(pts[0], pts[1]);
        for (int i = 1; i < 4; i++) path.lineTo(pts[2 * i], pts[2 * i + 1]);
        path.close();
        canvas.drawPath(path, paint);
    }
}
