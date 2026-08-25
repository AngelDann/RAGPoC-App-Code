package com.ragpoc.app

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.webkit.*
import android.widget.*
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var swipeRefresh: SwipeRefreshLayout
    private lateinit var webView: WebView
    private lateinit var loadingContainer: FrameLayout
    private lateinit var errorContainer: LinearLayout
    private lateinit var urlEditText: EditText
    private var fileChooserCallback: ValueCallback<Array<Uri>>? = null

    private val PREFS_NAME = "ragpoc_prefs"
    private val KEY_SERVER_URL = "server_url"
    private val DEFAULT_URL = "http://127.0.0.1:47823"

    private val filePickerLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (fileChooserCallback == null) return@registerForActivityResult
        val uris = WebChromeClient.FileChooserParams.parseResult(result.resultCode, result.data)
        fileChooserCallback?.onReceiveValue(uris)
        fileChooserCallback = null
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val rootLayout = FrameLayout(this).apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }

        // Swipe to Refresh wrapping WebView
        swipeRefresh = SwipeRefreshLayout(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        }

        webView = WebView(this).apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            visibility = View.GONE
        }
        swipeRefresh.addView(webView)
        rootLayout.addView(swipeRefresh)

        // Loading Spinner Container
        loadingContainer = FrameLayout(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
            setBackgroundColor(0xFFF8F9FA.toInt())
        }
        val progressBar = ProgressBar(this).apply { isIndeterminate = true }
        val pParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT
        ).apply { gravity = Gravity.CENTER }
        loadingContainer.addView(progressBar, pParams)

        val loadingText = TextView(this).apply {
            text = "Conectando a RAGPoC Studio…"
            textSize = 14f
            setTextColor(0xFF6C757D.toInt())
        }
        val tParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT
        ).apply {
            gravity = Gravity.CENTER
            topMargin = 140
        }
        loadingContainer.addView(loadingText, tParams)
        rootLayout.addView(loadingContainer)

        // Error & Server Config Container
        errorContainer = LinearLayout(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(48, 48, 48, 48)
            setBackgroundColor(0xFFFFFFFF.toInt())
            visibility = View.GONE
        }

        val titleView = TextView(this).apply {
            text = "RAGPoC · Knowledge Studio"
            textSize = 18f
            setTextColor(0xFF212529.toInt())
            gravity = Gravity.CENTER
        }
        val descView = TextView(this).apply {
            text = "Ingresa la dirección de tu servidor RAGPoC (o IP local / Tailscale):"
            textSize = 13f
            setTextColor(0xFF6C757D.toInt())
            setPadding(0, 16, 0, 24)
            gravity = Gravity.CENTER
        }
        urlEditText = EditText(this).apply {
            setText(getServerUrl())
            hint = "http://192.168.1.50:47823"
            textSize = 14f
            setPadding(24, 24, 24, 24)
            setBackgroundColor(0xFFF1F3F5.toInt())
        }
        val connectBtn = Button(this).apply {
            text = "Conectar"
            setOnClickListener {
                val inputUrl = urlEditText.text.toString().trim()
                if (inputUrl.isNotEmpty()) {
                    setServerUrl(inputUrl)
                    loadAppUrl(inputUrl)
                }
            }
        }
        val defBtn = Button(this).apply {
            text = "Usar Localhost (127.0.0.1:47823)"
            setOnClickListener {
                setServerUrl(DEFAULT_URL)
                urlEditText.setText(DEFAULT_URL)
                loadAppUrl(DEFAULT_URL)
            }
        }

        errorContainer.addView(titleView)
        errorContainer.addView(descView)
        errorContainer.addView(urlEditText)
        errorContainer.addView(connectBtn)
        errorContainer.addView(defBtn)
        rootLayout.addView(errorContainer)

        setContentView(rootLayout)

        setupWebView()
        setupBackNavigation()

        swipeRefresh.setOnRefreshListener {
            webView.reload()
        }

        loadAppUrl(getServerUrl())
    }

    private fun getServerUrl(): String {
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(KEY_SERVER_URL, DEFAULT_URL) ?: DEFAULT_URL
    }

    private fun setServerUrl(url: String) {
        val cleanUrl = if (!url.startsWith("http://") && !url.startsWith("https://")) {
            "http://$url"
        } else {
            url
        }
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().putString(KEY_SERVER_URL, cleanUrl).apply()
    }

    private fun loadAppUrl(url: String) {
        loadingContainer.visibility = View.VISIBLE
        errorContainer.visibility = View.GONE
        webView.visibility = View.GONE
        val targetUrl = if (!url.startsWith("http://") && !url.startsWith("https://")) "http://$url" else url
        webView.loadUrl(targetUrl)
    }

    private fun setupWebView() {
        val settings = webView.settings
        settings.javaScriptEnabled = true
        settings.domStorageEnabled = true
        settings.databaseEnabled = true
        settings.allowFileAccess = true
        settings.allowContentAccess = true
        settings.mediaPlaybackRequiresUserGesture = false
        settings.mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
        settings.useWideViewPort = true
        settings.loadWithOverviewMode = true

        webView.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                swipeRefresh.isRefreshing = false
                loadingContainer.visibility = View.GONE
                errorContainer.visibility = View.GONE
                webView.visibility = View.VISIBLE
            }

            override fun onReceivedError(
                view: WebView?,
                request: WebResourceRequest?,
                error: WebResourceError?
            ) {
                if (request?.isForMainFrame == true) {
                    swipeRefresh.isRefreshing = false
                    loadingContainer.visibility = View.GONE
                    webView.visibility = View.GONE
                    errorContainer.visibility = View.VISIBLE
                }
            }

            override fun shouldOverrideUrlLoading(
                view: WebView?,
                request: WebResourceRequest?
            ): Boolean {
                val url = request?.url?.toString() ?: return false
                val serverBase = getServerUrl()
                if (url.startsWith(serverBase) || url.startsWith("http://127.0.0.1") || url.startsWith("http://localhost")) {
                    return false
                }
                // External link -> open in device browser
                try {
                    val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
                    startActivity(intent)
                    return true
                } catch (_: Exception) {
                    return false
                }
            }
        }

        webView.webChromeClient = object : WebChromeClient() {
            override fun onShowFileChooser(
                view: WebView?,
                filePathCallback: ValueCallback<Array<Uri>>?,
                fileChooserParams: FileChooserParams?
            ): Boolean {
                fileChooserCallback?.onReceiveValue(null)
                fileChooserCallback = filePathCallback
                val intent = fileChooserParams?.createIntent() ?: Intent(Intent.ACTION_GET_CONTENT).apply {
                    type = "*/*"
                    addCategory(Intent.CATEGORY_OPENABLE)
                }
                try {
                    filePickerLauncher.launch(intent)
                } catch (_: Exception) {
                    fileChooserCallback = null
                    return false
                }
                return true
            }
        }
    }

    private fun setupBackNavigation() {
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (webView.canGoBack()) {
                    webView.goBack()
                } else {
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                }
            }
        })
    }

    fun installApkUpdate(apkFile: File) {
        try {
            val contentUri = FileProvider.getUriForFile(
                this,
                "$packageName.fileprovider",
                apkFile
            )
            val intent = Intent(Intent.ACTION_VIEW).apply {
                setDataAndType(contentUri, "application/vnd.android.package-archive")
                flags = Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_ACTIVITY_NEW_TASK
            }
            startActivity(intent)
        } catch (_: Exception) {}
    }
}
