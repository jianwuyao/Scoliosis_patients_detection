package org.jwyao.mnnpose;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.widget.Toast;

public class BootReceiver extends BroadcastReceiver {
    /**
     * This method is called when the BroadcastReceiver is receiving an Intent broadcast.
     * 开机自启动
     * @param context The Context in which the receiver is running.
     * @param intent The Intent being received.
     */
    @Override
    public void onReceive(Context context, Intent intent) {
        //Toast.makeText(context, intent.getAction(), Toast.LENGTH_LONG).show();
        Toast.makeText(context, "请稍候", Toast.LENGTH_LONG).show();
        String ACTION_BOOT = "android.intent.action.BOOT_COMPLETED";
        if (ACTION_BOOT.equals(intent.getAction())) {
            Intent intentMainActivity = new Intent(context, MainActivity.class);
            intentMainActivity.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            context.startActivity(intentMainActivity);
        }
    }
}
