// AI-on-Edges firmware — ESP32-S3, ESP-IDF + esp-tflite-micro.
//
// Role: a GENERIC model-runner. It simulates an ADC reading a mix of two secret
// frequencies, streams the raw signal, loads whatever .tflite the browser/builder
// pushes over MQTT, runs it, streams the two separated waveforms back, and — the
// point of the demo — reports precise on-device timings:
//   * load_us   : time to build the interpreter + AllocateTensors for a new model
//   * invoke_us : time for one window through the inference pipeline
//
// It never reveals the true frequencies; the browser must rediscover them by FFT.

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "freertos/semphr.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_random.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "mqtt_client.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "secrets.h"

static const char *TAG = "aiedge";

// ---- signal / model config (must match shared/architectures.json) ----------
#define FS_HZ 1024
#define N 256
static const int WINDOW_MS = 1000 * N / FS_HZ; // 250 ms
// Keep well below Nyquist (fs/2 = 512 Hz): near-Nyquist tones are only ~2 samples/cycle,
// which plots as spurious amplitude modulation. 200 Hz = ~5 samples/cycle.
static const float BAND_LO = 40.0f, BAND_HI = 200.0f;

// ---- MQTT topics -----------------------------------------------------------
#define T_ADC "adc/stream"
#define T_INFER "infer/device"
#define T_STATUS "status/device"
#define T_MODEL "model/flatbuffer"
#define T_RESHUFFLE "cmd/reshuffle"

static esp_mqtt_client_handle_t g_client = nullptr;

// ---- simulated ADC (the "secret" two-tone source) --------------------------
static volatile float g_f1 = 148.0f, g_f2 = 408.0f;
static float g_a1 = 0.8f, g_a2 = 0.6f;
static float g_ph1 = 0.0f, g_ph2 = 0.0f;  // running phases, wrapped to [0, 2pi) for float32 accuracy

static float frand(float lo, float hi) {
  return lo + (hi - lo) * (esp_random() / (float)UINT32_MAX);
}

static void reshuffle_freqs() {
  float f1, f2;
  do {
    f1 = frand(BAND_LO, BAND_HI);
    f2 = frand(BAND_LO, BAND_HI);
  } while (fabsf(f1 - f2) < 30.0f);
  if (f1 > f2) { float t = f1; f1 = f2; f2 = t; }
  g_f1 = f1; g_f2 = f2;
  ESP_LOGI(TAG, "reshuffle -> new secret freqs (hidden from the browser)");
}

static void gen_window(float *x) {
  // Phase-accumulator DDS: keep each sine argument in [0, 2pi) so float32 stays exact
  // indefinitely. sinf(w*f*idx) with an unbounded idx loses precision within minutes.
  const float two_pi = 2.0f * (float)M_PI;
  const float dph1 = two_pi * g_f1 / FS_HZ;  // per-sample phase step (snapshot; survives reshuffle)
  const float dph2 = two_pi * g_f2 / FS_HZ;
  for (int n = 0; n < N; n++) {
    float noise = frand(-1.0f, 1.0f) * 0.05f;
    x[n] = g_a1 * sinf(g_ph1) + g_a2 * sinf(g_ph2) + noise;
    g_ph1 += dph1; if (g_ph1 >= two_pi) g_ph1 -= two_pi;
    g_ph2 += dph2; if (g_ph2 >= two_pi) g_ph2 -= two_pi;
  }
}

// ---- TensorFlow Lite Micro --------------------------------------------------
alignas(16) static uint8_t g_arena[96 * 1024];
static uint8_t *g_model_buf = nullptr; // flatbuffer must outlive the interpreter
static tflite::MicroInterpreter *g_interp = nullptr;
static SemaphoreHandle_t g_model_mtx;

// Ops observed in the converted models (Conv1D stack; dilation -> space/batch).
static tflite::MicroMutableOpResolver<16> &get_resolver() {
  static tflite::MicroMutableOpResolver<16> r;
  static bool inited = false;
  if (!inited) {
    r.AddConv2D();
    r.AddQuantize();    // int8 models wrap an int8 core with float32 I/O
    r.AddDequantize();
    r.AddExpandDims();
    r.AddReshape();
    r.AddRelu();
    r.AddSpaceToBatchNd();
    r.AddBatchToSpaceNd();
    r.AddPad();
    r.AddDepthwiseConv2D();
    r.AddAdd();
    r.AddMul();
    r.AddFullyConnected();
    r.AddStridedSlice();
    inited = true;
  }
  return r;
}

static bool load_model(const uint8_t *data, size_t len) {
  int64_t t0 = esp_timer_get_time();
  uint8_t *buf = (uint8_t *)malloc(len);
  if (!buf) { ESP_LOGE(TAG, "OOM allocating %u-byte model", (unsigned)len); return false; }
  memcpy(buf, data, len);

  const tflite::Model *model = tflite::GetModel(buf);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "schema mismatch %lu != %d", (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
    free(buf);
    return false;
  }
  // Hold the lock across the WHOLE swap. AllocateTensors() writes into the shared g_arena,
  // so the old interpreter (which pipeline_task runs Invoke() on) must be torn down and idle
  // first. Building the new model while the old one is mid-Invoke corrupts the arena, and the
  // next Invoke jumps through a clobbered kernel pointer -> InstrFetchProhibited panic.
  xSemaphoreTake(g_model_mtx, portMAX_DELAY);
  if (g_interp) { delete g_interp; g_interp = nullptr; }
  auto *interp = new tflite::MicroInterpreter(model, get_resolver(), g_arena, sizeof(g_arena));
  if (interp->AllocateTensors() != kTfLiteOk) {
    delete interp;
    if (g_model_buf) { free(g_model_buf); g_model_buf = nullptr; }
    xSemaphoreGive(g_model_mtx);
    free(buf);
    ESP_LOGE(TAG, "AllocateTensors failed (arena too small?)");
    char err[96];
    snprintf(err, sizeof(err), "{\"event\":\"load_error\",\"tflite_bytes\":%u}", (unsigned)len);
    esp_mqtt_client_publish(g_client, T_STATUS, err, 0, 1, 0);
    return false;
  }
  if (g_model_buf) free(g_model_buf);
  g_interp = interp;
  g_model_buf = buf;
  size_t arena_used = interp->arena_used_bytes();
  xSemaphoreGive(g_model_mtx);

  int64_t load_us = esp_timer_get_time() - t0;
  char msg[192];
  snprintf(msg, sizeof(msg),
           "{\"event\":\"model_loaded\",\"load_us\":%lld,\"arena_bytes\":%u,\"tflite_bytes\":%u}",
           (long long)load_us, (unsigned)arena_used, (unsigned)len);
  esp_mqtt_client_publish(g_client, T_STATUS, msg, 0, 1, 0);
  ESP_LOGI(TAG, "model loaded: %u B, arena %u B, load %lld us",
           (unsigned)len, (unsigned)arena_used, (long long)load_us);
  return true;
}

// ---- JSON helpers -----------------------------------------------------------
static char g_json[9000]; // .bss, not stack

static int fmt_farray(char *out, int cap, const char *key, const float *a, int n) {
  int k = snprintf(out, cap, "\"%s\":[", key);
  for (int i = 0; i < n; i++) {
    if (k > cap - 16) break;
    k += snprintf(out + k, cap - k, "%s%.4f", i ? "," : "", a[i]);
  }
  k += snprintf(out + k, cap - k, "]");
  return k;
}

// ---- pipeline: generate -> stream -> infer -> stream ------------------------
static void pipeline_task(void *) {
  static float x[N], y0[N], y1[N];
  uint32_t seq = 0;
  int status_div = 0;

  while (true) {
    gen_window(x);
    seq++;

    // raw mixed signal
    int k = snprintf(g_json, sizeof(g_json), "{\"seq\":%u,\"fs\":%d,", (unsigned)seq, FS_HZ);
    k += fmt_farray(g_json + k, sizeof(g_json) - k, "x", x, N);
    k += snprintf(g_json + k, sizeof(g_json) - k, "}");
    esp_mqtt_client_publish(g_client, T_ADC, g_json, k, 0, 0); // qos 0: lossy telemetry, no outbox

    // inference (if a model has been loaded)
    bool did = false;
    int64_t invoke_us = 0;
    xSemaphoreTake(g_model_mtx, portMAX_DELAY);
    if (g_interp) {
      TfLiteTensor *in = g_interp->input(0);
      memcpy(in->data.f, x, N * sizeof(float));
      int64_t t0 = esp_timer_get_time();
      if (g_interp->Invoke() == kTfLiteOk) {
        invoke_us = esp_timer_get_time() - t0;
        TfLiteTensor *out = g_interp->output(0); // [1, N, 2] interleaved
        for (int i = 0; i < N; i++) { y0[i] = out->data.f[i * 2]; y1[i] = out->data.f[i * 2 + 1]; }
        did = true;
      }
    }
    xSemaphoreGive(g_model_mtx);

    if (did) {
      k = snprintf(g_json, sizeof(g_json), "{\"seq\":%u,\"invoke_us\":%lld,", (unsigned)seq, (long long)invoke_us);
      k += fmt_farray(g_json + k, sizeof(g_json) - k, "y0", y0, N);
      k += snprintf(g_json + k, sizeof(g_json) - k, ",");
      k += fmt_farray(g_json + k, sizeof(g_json) - k, "y1", y1, N);
      k += snprintf(g_json + k, sizeof(g_json) - k, "}");
      esp_mqtt_client_publish(g_client, T_INFER, g_json, k, 0, 0); // qos 0: lossy telemetry, no outbox

      if (++status_div >= 8) { // ~ every 2 s
        status_div = 0;
        char s[160];
        snprintf(s, sizeof(s), "{\"invoke_us\":%lld,\"free_heap\":%u}",
                 (long long)invoke_us, (unsigned)esp_get_free_heap_size());
        esp_mqtt_client_publish(g_client, T_STATUS, s, 0, 1, 0);
      }
    }
    vTaskDelay(pdMS_TO_TICKS(WINDOW_MS));
  }
}

// ---- MQTT -------------------------------------------------------------------
// esp-mqtt may fragment large payloads; reassemble using current_data_offset.
enum RxKind { RX_NONE, RX_MODEL, RX_RESHUFFLE };
static RxKind g_rx_kind = RX_NONE;
static uint8_t *g_rx = nullptr;
static int g_rx_cap = 0;

static bool topic_is(esp_mqtt_event_handle_t e, const char *t) {
  return e->topic && (int)strlen(t) == e->topic_len && strncmp(e->topic, t, e->topic_len) == 0;
}

static void mqtt_event_handler(void *, esp_event_base_t, int32_t event_id, void *event_data) {
  esp_mqtt_event_handle_t e = (esp_mqtt_event_handle_t)event_data;
  switch ((esp_mqtt_event_id_t)event_id) {
  case MQTT_EVENT_CONNECTED:
    ESP_LOGI(TAG, "mqtt connected");
    esp_mqtt_client_subscribe(g_client, T_MODEL, 1);
    esp_mqtt_client_subscribe(g_client, T_RESHUFFLE, 1);
    break;
  case MQTT_EVENT_DATA:
    if (e->current_data_offset == 0) { // first fragment carries the topic
      g_rx_kind = topic_is(e, T_MODEL) ? RX_MODEL : topic_is(e, T_RESHUFFLE) ? RX_RESHUFFLE : RX_NONE;
      if (g_rx_kind != RX_NONE && g_rx_cap < e->total_data_len) {
        free(g_rx);
        g_rx = (uint8_t *)malloc(e->total_data_len);
        g_rx_cap = g_rx ? e->total_data_len : 0;
      }
    }
    if (g_rx_kind != RX_NONE && g_rx) {
      memcpy(g_rx + e->current_data_offset, e->data, e->data_len);
      if (e->current_data_offset + e->data_len >= e->total_data_len) {
        if (g_rx_kind == RX_MODEL) load_model(g_rx, e->total_data_len);
        else if (g_rx_kind == RX_RESHUFFLE) reshuffle_freqs();
        g_rx_kind = RX_NONE;
      }
    }
    break;
  case MQTT_EVENT_ERROR:
    ESP_LOGW(TAG, "mqtt error");
    break;
  default:
    break;
  }
}

static void mqtt_start() {
  esp_mqtt_client_config_t cfg = {};
  cfg.broker.address.uri = MQTT_URI;
  cfg.buffer.size = 20480;     // hold a whole model in one message
  cfg.buffer.out_size = 12288; // hold a whole infer/device JSON
  // The MQTT event task calls load_model() -> AllocateTensors(); give it headroom.
  cfg.task.stack_size = 16384;
  cfg.outbox.limit = 16 * 1024; // cap unacked-QoS1 backlog so a broker/network stall can't exhaust heap
  g_client = esp_mqtt_client_init(&cfg);
  esp_mqtt_client_register_event(g_client, MQTT_EVENT_ANY, mqtt_event_handler, nullptr);
  esp_mqtt_client_start(g_client);
  ESP_LOGI(TAG, "mqtt connecting to %s", MQTT_URI);
}

// ---- Wi-Fi station ----------------------------------------------------------
static EventGroupHandle_t s_wifi_eg;
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT BIT1
static int s_retry = 0;

static void wifi_evt(void *, esp_event_base_t base, int32_t id, void *data) {
  if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
    esp_wifi_connect();
  } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
    if (s_retry < 100) { esp_wifi_connect(); s_retry++; ESP_LOGI(TAG, "wifi retry %d", s_retry); }
    else xEventGroupSetBits(s_wifi_eg, WIFI_FAIL_BIT);
  } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
    ip_event_got_ip_t *ev = (ip_event_got_ip_t *)data;
    ESP_LOGI(TAG, "got ip " IPSTR, IP2STR(&ev->ip_info.ip));
    s_retry = 0;
    xEventGroupSetBits(s_wifi_eg, WIFI_CONNECTED_BIT);
  }
}

static void wifi_init_sta() {
  s_wifi_eg = xEventGroupCreate();
  esp_netif_create_default_wifi_sta();
  wifi_init_config_t ic = WIFI_INIT_CONFIG_DEFAULT();
  ESP_ERROR_CHECK(esp_wifi_init(&ic));
  esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_evt, nullptr, nullptr);
  esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_evt, nullptr, nullptr);

  wifi_config_t wc = {};
  strncpy((char *)wc.sta.ssid, WIFI_SSID, sizeof(wc.sta.ssid) - 1);
  strncpy((char *)wc.sta.password, WIFI_PASS, sizeof(wc.sta.password) - 1);
  ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
  ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
  ESP_ERROR_CHECK(esp_wifi_start());
  ESP_LOGI(TAG, "wifi connecting to \"%s\" ...", WIFI_SSID);
  xEventGroupWaitBits(s_wifi_eg, WIFI_CONNECTED_BIT | WIFI_FAIL_BIT, pdFALSE, pdFALSE, portMAX_DELAY);
}

extern "C" void app_main(void) {
  ESP_ERROR_CHECK(nvs_flash_init());
  ESP_ERROR_CHECK(esp_netif_init());
  ESP_ERROR_CHECK(esp_event_loop_create_default());
  g_model_mtx = xSemaphoreCreateMutex();
  reshuffle_freqs();

  wifi_init_sta();
  mqtt_start();
  xTaskCreate(pipeline_task, "pipeline", 16384, nullptr, 5, nullptr);
  ESP_LOGI(TAG, "up: fs=%d N=%d window=%dms", FS_HZ, N, WINDOW_MS);
}
