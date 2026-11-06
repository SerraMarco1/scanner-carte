package it.scannercarte.app;

import android.os.Bundle;

import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        registerPlugin(ScannerPlugin.class); // modulo nativo dello scanner, prima di avviare la pagina
        super.onCreate(savedInstanceState);
    }
}
