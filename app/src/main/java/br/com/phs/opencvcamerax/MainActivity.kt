package br.com.phs.opencvcamerax

import android.Manifest.permission.CAMERA
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.graphics.*
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import br.com.phs.opencvcamerax.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import java.io.BufferedInputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

@ExperimentalGetImage
class MainActivity : AppCompatActivity() {

    private lateinit var viewBinding: ActivityMainBinding
    private var imageCapture: ImageCapture? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var converter: YuvToRgbConverter
    private var net: Net? = null
    private var busy = false

    private val classNames = arrayOf(
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"
    )

    private val loaderCallBack = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when(status) {
                SUCCESS -> {
                    Log.i(TAG, "OpenCV loaded successfully")
                    this@MainActivity.startCamera()
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        this.viewBinding.viewFinder.setOnClickListener {

            if (busy) return@setOnClickListener

            this.viewBinding.imageProcessProgressBar.visibility = View.VISIBLE
            this.takeImageAnalyzer()
            //this.takePictureAndSave()
            //this.takePicture()
        }

        this.viewBinding.posProcessImgCloseBtn.setOnClickListener {
            this.viewBinding.posProcessImg.visibility = View.GONE
            this.viewBinding.posProcessImgCloseBtn.visibility = View.GONE
            runBlocking {
                withContext(Dispatchers.IO) {
                    Thread.sleep(200)
                    startCamera()
                }
            }
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        this.hideSystemBars()
        this.initConverter()

    }

    private fun hideSystemBars() {
        val windowInsetsController = ViewCompat.getWindowInsetsController(window.decorView) ?: return
        // Configure the behavior of the hidden system bars
        windowInsetsController.systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        // Hide both the status bar and the navigation bar
        windowInsetsController.hide(WindowInsetsCompat.Type.systemBars())
    }

    private fun initConverter() {
        converter = YuvToRgbConverter(this)
    }

    private fun onCameraViewStarted() {

        val proto: String = getPath("MobileNetSSD_deploy.prototxt", this)
        val weights: String = getPath("MobileNetSSD_deploy.caffemodel", this)
        net = Dnn.readNetFromCaffe(proto, weights)
        Log.i(TAG, "Network loaded successfully")

    }

    private fun takePictureAndSave() {

        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return

        // Create time stamped name and MediaStore entry.
        val name = SimpleDateFormat(FILENAME_FORMAT, Locale.US)
            .format(System.currentTimeMillis())
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if(Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/CameraX-Image")
            }
        }

        // Create output options object which contains file + metadata
        val outputOptions = ImageCapture.OutputFileOptions
            .Builder(contentResolver,
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues)
            .build()

        // Set up image capture listener, which is triggered after photo has
        // been taken
        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults){
                    val msg = "Photo capture succeeded: ${output.savedUri}"
                    Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                    Log.d(TAG, msg)
                }
            }
        )

    }

    @SuppressLint("RestrictedApi")
    private fun takeImageAnalyzer() {

        imageAnalyzer?.setAnalyzer(cameraExecutor) { image ->

            this.busy = true
            var rotatedBitmapImage: Bitmap? = null
            var rgba: Mat? = null

            image.image?.let {

                val bitmapImage = Bitmap.createBitmap(
                    image.width, image.height,
                    Bitmap.Config.ARGB_8888
                )
                converter.yuvToRgb(it, bitmapImage!!)
                rotatedBitmapImage = bitmapImage.rotateBitmapPhoto(image.imageInfo.rotationDegrees.toFloat())
                rgba = image.image?.yuvToRgba()

            }

            imageAnalyzer?.clearAnalyzer()
            image.close()

            this.onCameraViewStarted()
            val list = this.processImage(rgba)

            if (list.isNotEmpty()) {
                this.drawRectangles(rotatedBitmapImage, list)
            } else {
                runOnUiThread {
                    this.viewBinding.imageProcessProgressBar.visibility = View.GONE
                }
                this.startCamera()
            }

        }

    }

    private fun processImage(frame: Mat?): List<Pair<String, RectF>> {

        frame?: return emptyList()

        val rectList = mutableListOf<Pair<String, RectF>>()

        val inWidth = 300
        val inHeight = 300
        val inScaleFactor = 0.007843
        val meanVal = 127.5
        val threshold = 0.5
        // Get a new frame

        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB)
        // Forward image through network.

        val blob = Dnn.blobFromImage(
            frame, inScaleFactor,
            Size(inWidth.toDouble(), inHeight.toDouble()),
            Scalar(meanVal, meanVal, meanVal), false, false
        )

        net?.setInput(blob)
        var detections = net?.forward()?: return emptyList()
        val cols = frame.cols()
        val rows = frame.rows()
        detections = detections.reshape(1, detections.total().toInt() / 7)

        for (i in 0 until detections.rows()) {

            val confidence = detections[i, 2][0]
            if (confidence > threshold) {
                val classId = detections[i, 1][0].toInt()
                val left = (detections[i, 3][0] * cols)
                val top = (detections[i, 4][0] * rows)
                val right = (detections[i, 5][0] * cols)
                val bottom = (detections[i, 6][0] * rows)

                val rect = RectF(left.toFloat(), top.toFloat(), right.toFloat(), bottom.toFloat())

                // Draw rectangle around detected object.
                Imgproc.rectangle(
                    frame, Point(left, top), Point(right, bottom),
                    Scalar(0.0, 255.0, 0.0)
                )

                if (classId >= 0 && classId < classNames.size) {

                    val label = classNames[classId] + ": " + confidence
                    rectList.add(Pair(label, rect))
                    Log.i("Detected: ", label)
                    val baseLine = IntArray(1)
                    val labelSize: Size =
                        Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine)
                    // Draw background for label.
                    Imgproc.rectangle(
                        frame, Point(left, top - labelSize.height),
                        Point(left + labelSize.width, top + baseLine[0]),
                        Scalar(255.0, 255.0, 255.0), Imgproc.FILLED
                    )
                    // Write class name and confidence.
                    Imgproc.putText(frame, label, Point(left, top),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0.0, 0.0, 0.0)
                    )

                }

            }

        }

        return rectList

    }

    private fun drawRectangles(bt: Bitmap?, list: List<Pair<String, RectF>>) {

        bt?: return

        val scale = resources.displayMetrics.density

        val tempBitmap = Bitmap.createScaledBitmap(bt, bt.width, bt.height, true)
        val canvas = Canvas(tempBitmap)
        val p = Paint()
        p.style = Paint.Style.STROKE
        p.strokeWidth = 2f
        p.isAntiAlias = true
        p.isFilterBitmap = true
        p.isDither = true
        p.color = Color.RED
        // Text
        val textPaint = Paint()
        textPaint.style = Paint.Style.FILL_AND_STROKE
        textPaint.color = Color.RED
        textPaint.textSize = (6 * scale)

        list.forEach { rect ->

            // Draw Rectangle
            canvas.drawRect(rect.second, p)
            // Draw Label
            val bounds = Rect()
            textPaint.getTextBounds(rect.first, 0, rect.first.length, bounds)
            canvas.drawText(rect.first, rect.second.left, rect.second.top, textPaint)

        }

        runOnUiThread {
            this.viewBinding.posProcessImg.visibility = View.VISIBLE
            this.viewBinding.posProcessImgCloseBtn.visibility = View.VISIBLE
            this.viewBinding.posProcessImg.setImageBitmap(tempBitmap)
            this.viewBinding.imageProcessProgressBar.visibility = View.GONE
        }

    }

    // Upload file to storage and return a path.
    private fun getPath(file: String, context: Context): String {

        val assetManager: AssetManager = context.assets
        val inputStream: BufferedInputStream?

        try {

            // Read data from assets.
            inputStream = BufferedInputStream(assetManager.open(file))
            val data = ByteArray(inputStream.available())
            inputStream.read(data)
            inputStream.close()
            // Create copy file in storage.
            val outFile = File(context.filesDir, file)
            val os = FileOutputStream(outFile)
            os.write(data)
            os.close()
            // Return a path to file which may be read in common way.
            return outFile.absolutePath

        } catch (ex: IOException) {
            Log.i(TAG, "Failed to upload a file")
        }

        return ""

    }

    private fun takePicture() {

        imageCapture?.takePicture(cameraExecutor, object : ImageCapture.OnImageCapturedCallback() {

                override fun onCaptureSuccess(image: ImageProxy) {
                    super.onCaptureSuccess(image)

                    image.let {
                        val bitmap = it.imageProxyToBitmap()
                        takeImageAnalyzer()
                        it.close()
                    }

                }

                override fun onError(exception: ImageCaptureException) {
                    super.onError(exception)
                    Log.e("takePicture", exception.message?: "UNKNOWN")
                }

            }

        )

    }

    private fun openCvInitialization() {

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, loaderCallBack)
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!")
            loaderCallBack.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }

    }

    private fun startCamera() {

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val rotation = viewBinding.viewFinder.display.rotation

            // Preview
            val preview = Preview.Builder()
                .setTargetRotation(rotation)
                .build()
                .also {
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder().build()

            imageAnalyzer = ImageAnalysis.Builder().build()

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {

                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalyzer)

            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))

        this.busy = false

    }

    override fun onResume() {
        super.onResume()

        if (ContextCompat.checkSelfPermission(this, CAMERA) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, arrayOf(CAMERA), PERMISSION_REQUEST_CAMERA)
        } else {
            this.openCvInitialization()
        }

    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {

        private const val PERMISSION_REQUEST_CAMERA = 0
        private const val TAG = "MainActivity"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"

    }

}