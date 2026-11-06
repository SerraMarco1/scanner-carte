package it.scannercarte.app;

import android.os.Bundle;

import com.getcapacitor.BridgeActivity;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        // Se l'app si chiude per un errore, lo scrive in crash.txt: al riavvio la pagina lo mostra nei Risultati.
        Thread.UncaughtExceptionHandler previous = Thread.getDefaultUncaughtExceptionHandler();
        Thread.setDefaultUncaughtExceptionHandler((thread, error) -> {
            try (PrintWriter w = new PrintWriter(new FileWriter(new File(getFilesDir(), "crash.txt")))) {
                error.printStackTrace(w);
            } catch (Exception ignored) { /* niente da fare: l'app si sta chiudendo */ }
            if (previous != null) previous.uncaughtException(thread, error);
        });
        registerPlugin(ScannerPlugin.class); // modulo nativo dello scanner, prima di avviare la pagina
        super.onCreate(savedInstanceState);
    }
}
