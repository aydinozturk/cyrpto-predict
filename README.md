# cryptopredict

> ## ⚠️ YATIRIM TAVSİYESİ DEĞİLDİR
>
> Bu proje **eğitim ve araştırma amaçlıdır**. Üretilen tahminler, metrikler ve
> backtest sonuçları hiçbir şekilde alım-satım önerisi değildir. Kripto paralar
> çok oynaktır; geçmiş performans gelecekteki sonuçları garanti etmez. Bu kodla
> verdiğiniz her kararın sorumluluğu size aittir.

Kripto para fiyatları için **log-getiri tahmini**: Binance'ten OHLCV mumlarını
çeker, sızıntısız (look-ahead'siz) teknik özellikler üretir, klasik ML modelleri
ve basit baseline'larla `h` bar sonrasının getirisini tahmin eder; sonuçları
**walk-forward** doğrulama ve işlem maliyetli bir **backtest** ile değerlendirir.

Amaç "fiyatı bilmek" değil, bir modelin naif baseline'lardan (ör. "getiri = 0")
gerçekten daha iyi olup olmadığını **dürüst** biçimde ölçmektir.

## İçindekiler

- [Kurulum](#kurulum)
- [Hızlı başlangıç (CLI)](#hızlı-başlangıç-cli)
- [Binance erişimi](#binance-erişimi)
- [Mimari](#mimari)
- [Hedef değişken ve özellikler](#hedef-değişken-ve-özellikler)
- [Modeller](#modeller)
- [Değerlendirme yöntemi](#değerlendirme-yöntemi)
- [Metrikler](#metrikler)
- [Python API](#python-api)
- [Testler](#testler)
- [Sınırlamalar](#sınırlamalar)
- [English summary](#english-summary)

## Kurulum

Python **3.11+** gerekir.

```bash
git clone https://github.com/aydinozturk/cyrpto-predict.git
cd cyrpto-predict
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Bağımlılıklar: pandas, numpy, scikit-learn, requests, joblib (geliştirme için pytest).
Kurulumdan sonra `cryptopredict` komutu kullanılabilir (`python -m cryptopredict` ile aynı).

## Hızlı başlangıç (CLI)

```bash
# 1) Veriyi çek ve önbelleğe yaz
cryptopredict fetch --symbol BTCUSDT --interval 1h --start 2024-01-01 --cache-dir data/

# 2) Model eğit ve kaydet
cryptopredict train --symbol BTCUSDT --interval 1h --start 2024-01-01 \
    --model ridge --horizon 1 --out models/btc_ridge.joblib

# 3) Son kapanmış mum için tahmin
cryptopredict predict --model-path models/btc_ridge.joblib

# 4) Walk-forward değerlendirme + backtest (tüm modelleri karşılaştır)
cryptopredict backtest --symbol BTCUSDT --interval 1h --start 2024-01-01 \
    --model all --splits 5 --fee-bps 10
```

Ağ erişimi olmadan denemek için her komut `--csv PATH` ile yerel bir OHLCV
dosyası kabul eder; repodaki örnek veri (600 saatlik BTCUSDT mumu):

```bash
CSV=tests/fixtures/btcusdt_1h_sample.csv
cryptopredict backtest --csv $CSV --model all --splits 5
cryptopredict train    --csv $CSV --model ridge --horizon 1 --out models/x.joblib
cryptopredict predict  --csv $CSV --model-path models/x.joblib --json
```

`--csv` ile verilen dosyanın son satırı kapanmış mum kabul edilir.

### Komutlar ve bayraklar

**Ortak veri bayrakları** (`fetch`, `train`, `predict`, `backtest`):

| Bayrak | Açıklama |
|---|---|
| `--symbol BTCUSDT` | Binance sembolü (varsayılan `BTCUSDT`) |
| `--interval 1h` | Mum aralığı (`1m`, `5m`, `15m`, `1h`, `4h`, `1d`, …; varsayılan `1h`) |
| `--start 2024-01-01` | Başlangıç tarihi (UTC); `--csv` yoksa zorunlu (`predict` hariç) |
| `--end YYYY-MM-DD` | Bitiş tarihi (opsiyonel; varsayılan: şimdi) |
| `--cache-dir data/` | Çekilen mumların önbellek dizini (varsayılan `data/`; dosya `data/BTCUSDT_1h.csv`) |
| `--csv PATH` | Binance yerine yerel CSV kullan (ağsız) |
| `--json` | Makine okunur JSON çıktı (tüm komutlarda) |

| Komut | Ek bayraklar | Ne yapar |
|---|---|---|
| `fetch` | `--out PATH.csv` | Mumları çeker, önbelleğe (ve isteğe bağlı `--out`'a) yazar, özet basar |
| `train` | `--model NAME --horizon H --out PATH.joblib` | Tüm veriyle modeli eğitir; model, özellik yapılandırması, sembol/aralık ve eğitim dönemiyle birlikte kaydeder (varsayılan model `ridge`, horizon `1`) |
| `predict` | `--model-path PATH [--csv PATH]` | Son **kapanmış** mum için tahmini log-getiri, beklenen fiyat, yön (yukarı/aşağı) ve horizon'u basar. `--csv` yoksa modelin metadata'sındaki sembol/aralık için son ~500 bar çekilir (CSV ile eğitilmiş modelde `--symbol` verin) |
| `backtest` | `--model NAME\|all --horizon H --splits K --fee-bps F [--slippage-bps S] [--threshold T] [--da-threshold D] [--allow-short]` | Walk-forward kat metrikleri, genel metrikler ve strateji vs. buy-and-hold; `all` ile tüm modelleri RMSE'ye göre sıralı bir tabloda karşılaştırır. Varsayılanlar: `--splits 5`, `--fee-bps 10`, `gap = horizon`. `--threshold` yalnız pozisyon eşiğidir (`\|tahmin\| > T` ise işlem); `--da-threshold` yön isabetindeki nötr banttır (`\|gerçek\| <= D` olan barlar hariç). İkisinin de varsayılanı 0 |

Model adları: `zero`, `mean`, `last`, `ma`, `ridge`, `gbm`.

`predict --json` çıktısının anahtarları: `model`, `symbol`, `interval`, `horizon`,
`as_of` (son kapanmış mumun `open_time`'ı), `target_time` (tahmin edilen
kapanışın zamanı: `as_of + (horizon + 1) × aralık`), `last_close`,
`predicted_log_return`, `predicted_pct_change` (yüzde), `expected_price`
(`last_close * exp(predicted_log_return)`) ve `direction` (`up` / `down` / `flat`).

CSV biçimi: `open_time` (UTC zaman damgası) ve `open,high,low,close,volume` kolonları.

## Binance erişimi

Veri, Binance'in herkese açık REST uç noktasından (`/api/v3/klines`, API anahtarı
gerekmez) çekilir. Varsayılan adres `https://api.binance.com`'dur; farklı bir
adres (ör. `https://data-api.binance.vision` aynası) ortam değişkeniyle verilebilir:

```bash
export CRYPTOPREDICT_BINANCE_URL=https://data-api.binance.vision
```

Ortam değişkeni `cryptopredict.data` içe aktarılırken okunur
(`DEFAULT_BASE_URL`); Python API'de istek başına `base_url=` parametresiyle de
verilebilir (`fetch_klines`, `get_ohlcv`). Yalnızca **kapanmış** mumlar döner,
istekler 1000'lik sayfalarla ilerler ve 418/429/5xx yanıtlarında yeniden denenir.
Binance geçmişinde bakım/kesinti kaynaklı gerçek mum boşlukları vardır (ör.
2023-03-24); bunlar varsayılan olarak uyarı verir (`on_gap="warn"`), `"raise"`
hata fırlatır, `"ignore"` sessiz geçer.

Önbellek: `get_ohlcv` mumları `<cache_dir>/<SYMBOL>_<interval>.csv` dosyasında
tutar (ör. `data/BTCUSDT_1h.csv`) ve yalnızca eksik kalan aralıkları çeker.

> **HTTP 451 notu:** Binance bazı ülkelerden/bulut bölgelerinden (ör. ABD) gelen
> istekleri coğrafi olarak engeller ve `451 Unavailable For Legal Reasons` döner.
> Bu durumda `CRYPTOPREDICT_BINANCE_URL` ile erişilebilir bir aynayı kullanın ya da
> veriyi başka yoldan indirip `--csv` ile verin.

## Mimari

```
cryptopredict/
├── core/        # Ortak sözleşme: OHLCV şeması, validate_ohlcv, Forecaster protokolü, sabitler
├── data/        # Binance klines istemcisi, sayfalama, disk önbelleği, CSV okuma
├── features/    # FeatureConfig, teknik göstergeler, hedef (target), make_dataset
├── models/      # Baseline'lar, Ridge/GBM, model kaydı (registry), kaydet/yükle
├── evaluation/  # walk_forward_splits, metrikler, walk_forward_evaluate, backtest
├── pipeline.py  # Uçtan uca akış (CLI'ın kullandığı yardımcılar)
└── cli.py       # `cryptopredict` komut satırı
```

Veri akışı:

```
Binance REST / CSV
        │  data
        ▼
OHLCV DataFrame  (index: open_time UTC, kolonlar: open high low close volume)
        │  core.validate_ohlcv
        ▼
features.make_dataset ──► X (özellikler, t anına kadarki bilgi)
        │                 y (target = log(close[t+h] / close[t]))
        ▼
models.get_model(name) ──► fit(X, y) / predict(X)
        │
        ├─► evaluation.walk_forward_evaluate ─► kat ve genel metrikler
        │                                         │
        │                                         ▼
        │                          evaluation.backtest ─► strateji vs. buy&hold
        │
        └─► models.save_model / load_model ─► CLI predict
```

`core/types.py` tüm alt paketlerin dayandığı sözleşmedir: `INDEX_NAME = "open_time"`,
`OHLCV_COLUMNS`, `TARGET_COLUMN = "target"`, `LOG_RET_1 = "log_ret_1"`,
`DEFAULT_HORIZON = 1`, `validate_ohlcv()` ve `Forecaster` protokolü
(`fit(X, y) -> self`, `predict(X) -> np.ndarray`).

## Hedef değişken ve özellikler

**Hedef:** `h` bar sonrasının log-getirisi

```
target[t] = log(close[t + h] / close[t])        (varsayılan h = 1)
```

Fiyat yerine log-getiri tahmin edilir: getiriler fiyattan çok daha durağandır,
farklı fiyat seviyelerinde karşılaştırılabilirdir ve toplanabilir. Tahmini fiyat
gerekirse `close[t] * exp(tahmin)` ile geri çevrilir.

**Özellikler** (`FeatureConfig` varsayılanları; yalnızca `t` anı ve öncesindeki
veriyi kullanır):

| Kolon | Anlam |
|---|---|
| `log_ret_1`, `log_ret_2`, `log_ret_3`, `log_ret_6`, `log_ret_12`, `log_ret_24` | Son N barın log-getirisi `log(close[t] / close[t-N])` (`log_ret_1` her zaman vardır) |
| `ret_mean_24` | Son 24 barın ortalama 1-bar log-getirisi |
| `close_sma_6_ratio`, `close_sma_12_ratio`, `close_sma_24_ratio` | Kapanışın basit hareketli ortalamadan göreli sapması: `close / SMA − 1` |
| `close_ema_6_ratio`, `close_ema_12_ratio`, `close_ema_24_ratio` | Kapanışın üstel hareketli ortalamadan göreli sapması: `close / EMA − 1` |
| `rsi_14` | 14 barlık RSI (Wilder yumuşatması, 0–100) |
| `macd`, `macd_signal` | MACD (EMA12 − EMA26) ve sinyal çizgisi (EMA9); ham fiyat biriminde |
| `bollinger_pct_b_20`, `bollinger_width_20` | Bollinger bandı (20, 2σ) içindeki konum ve bant genişliği |
| `volatility_24` | Son 24 barın 1-bar log-getiri standart sapması |
| `volume_zscore_24` | Hacmin 24 barlık z-skoru |
| `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` | Saat / haftanın günü (yalnızca `add_time_features=True` ise) |

Kolon adları pencere parametrelerinden türetilir (ör. `rsi_window=21` → `rsi_21`).
Varsayılanlar `FeatureConfig(...)` ya da kısmi bir sözlükle değiştirilebilir:
`make_dataset(ohlcv, horizon=1, config={"rsi_window": 21, "add_time_features": True})`.
`FeatureConfig.to_dict()` / `FeatureConfig.from_dict()` yapılandırmayı model
metadata'sına yazıp tahminde aynı özellikleri yeniden üretmeye yarar.

- `features.build_features(df, config)` — tüm özellikler, `df` ile aynı indeks;
  ısınma (warmup) satırları NaN kalır.
- `features.make_dataset(df, horizon, config)` → `(X, y)`: ısınma satırları ve
  hedefi henüz bilinmeyen son `h` satır atılır; `X` ve `y` aynı indeksi paylaşır,
  NaN/sonsuz değer içermez (impute yapılmaz).
- `features.latest_features(df, config)` — canlı tahmin için son eksiksiz satır.
- `features.forward_log_return(close, horizon)` / `features.make_target(df, horizon)` — hedef serisi.

## Modeller

| Ad | Sınıf | Tahmin |
|---|---|---|
| `zero` | `ZeroReturn` | Her zaman 0 (rastgele yürüyüş: "fiyat değişmez") — ana baseline |
| `mean` | `MeanReturn` | Eğitim hedeflerinin ortalaması (sabit drift) |
| `last` | `LastReturn` | Son gözlenen getiri, `log_ret_1` (momentum) |
| `ma` | `MovingAverageReturn` | Son `window` (24) eğitim hedefinin ortalaması; `column="ret_mean_24"` verilirse o kolon (CLI bunu kullanır) |
| `ridge` | `RidgeForecaster` | StandardScaler + Ridge regresyon (`alpha=1.0`) |
| `gbm` | `GBMForecaster` | HistGradientBoostingRegressor (deterministik, `random_state=42`) |

Tüm modeller scikit-learn `BaseEstimator` tabanlıdır (`sklearn.base.clone`
çalışır). Ölçekleme pipeline'ın içindedir, yani walk-forward'da yalnızca eğitim
penceresine fit edilir. Model `models.get_model("ridge", alpha=10.0)` ile
oluşturulur; `models.available_models()` kayıtlı adları döner.

## Değerlendirme yöntemi

Zaman serilerinde rastgele train/test bölmesi geleceği eğitime sızdırır. Bu
yüzden **walk-forward** (ileriye yürüyen) doğrulama kullanılır:

```
kat 0: [==== train ====]  gap  [test]
kat 1: [====== train ======]  gap  [test]
kat 2: [======== train ========]  gap  [test]
                                           → zaman
```

- Test blokları ardışık, çakışmasız ve serinin sonundadır; her katta eğitim
  verisi test verisinden **kesinlikle önce** gelir.
- Her katta **yeni, eğitilmemiş** bir model kurulur (`model_factory`), yalnızca o
  katın eğitim satırlarını görür.
- `expanding=True` (varsayılan) büyüyen pencere, `expanding=False` sabit
  `train_size` uzunluğunda kayan pencere kullanır.
- **gap = horizon:** `target[t]` ancak `t + h` anında bilinir; eğitim ile test
  arasına `gap` örnek boşluk bırakmak, eğitim hedeflerinin test dönemine taşmasını
  engeller (`gap >= h - 1` zorunlu, `gap = h` güvenli seçim; CLI bunu kullanır).

Sızıntı önlemleri özetle: özellikler yalnızca geçmiş/mevcut barı kullanır
(merkezli pencere ya da `shift(-k)` yok), ölçekleyici her katta yeniden fit
edilir, gap uygulanır ve backtest bir sonraki barın getirisini kullanır.

## Metrikler

Nokta tahmin metrikleri (`evaluation.regression_report`):

| Metrik | Anlam |
|---|---|
| `mae` | Ortalama mutlak hata — log-getiri biriminde (0.001 ≈ %0.1) |
| `rmse` | Kök ortalama kare hata — büyük hataları daha çok cezalandırır |
| `smape` | Simetrik MAPE, yüzde (0–200). Getiriler 0'a yakın olduğundan oynaktır; dikkatli yorumlayın |
| `directional_accuracy` | Yön isabeti: `sign(tahmin) == sign(gerçek)` oranı. `\|gerçek\|` değeri `--da-threshold` altında kalan küçük hareketler hariç tutulur (varsayılan 0). 0.5 ≈ yazı-tura; `zero` hiç yön tahmin etmediği için 0 alır |
| `relative_mae` | `MAE(model) / MAE(zero baseline)`. **< 1 ise model "getiri = 0" baseline'ını yener**; ≥ 1 ise yenemez |

Backtest (`evaluation.backtest`) tahminin işaretiyle işlem yapar: tahmin
`> threshold` → long (+1), `< -threshold` ve `allow_short` → short (−1), aksi halde
nakitte (0). Pozisyon `t` kapanışında açılıp `t+1` kapanışına kadar tutulur ve
**ileri 1-bar getiriyi** (h=1 `target`, geriye bakan `log_ret_1` değil) kazanır.

| İstatistik | Anlam |
|---|---|
| `total_return` / `bh_total_return` | Strateji / buy-and-hold toplam getirisi (maliyetler düşülmüş) |
| `excess_return` | Strateji − buy-and-hold |
| `sharpe` / `bh_sharpe` | Yıllıklandırılmış Sharpe oranı (risksiz faiz 0, varsayılan saatlik bar: `periods_per_year = 24*365`) |
| `max_drawdown` / `bh_max_drawdown` | En büyük tepe-dip kaybı (pozitif oran, 0.25 = %25) |
| `n_trades` | Pozisyon değişikliği sayısı |
| `exposure` | Piyasada geçirilen bar oranı |
| `total_cost` | Toplam işlem maliyeti |

Maliyet: her pozisyon değişikliğinde `|Δpozisyon| × (fee_bps + slippage_bps) / 10 000`.
`--fee-bps 10` = işlem başına %0.1 komisyon. Buy-and-hold bir kez giriş maliyeti öder.

## Python API

Örnek repodaki fixture CSV ile ağsız çalışır:

```python
from cryptopredict.data import load_csv
from cryptopredict.evaluation import backtest, walk_forward_evaluate
from cryptopredict.features import make_dataset
from cryptopredict.models import get_model

ohlcv = load_csv("tests/fixtures/btcusdt_1h_sample.csv")
# Canlı veri: from cryptopredict.data import get_ohlcv
# ohlcv = get_ohlcv("BTCUSDT", "1h", start="2024-01-01", cache_dir="data/")

horizon = 1
X, y = make_dataset(ohlcv, horizon=horizon)

for name in ["zero", "last", "ridge", "gbm"]:
    res = walk_forward_evaluate(lambda: get_model(name), X, y, n_splits=5, gap=horizon)
    # h=1'de y_true = ileri 1-bar getiri; h>1 için forward_log_return(close, 1) kullanın
    bt = backtest(res.predictions["y_pred"], res.predictions["y_true"], fee_bps=10)
    print(f"{name:6s} relative_mae={res.overall['relative_mae']:.3f} "
          f"yön={res.overall['directional_accuracy']:.3f} "
          f"getiri={bt.stats['total_return']:+.2%} (b&h {bt.stats['bh_total_return']:+.2%})")
```

`res.folds` kat başına metrikleri, `res.predictions` (`y_true`, `y_pred`, `fold`)
örneklem dışı tahminleri, `bt.frame` bar bazında pozisyon/maliyet/özsermaye
eğrisini içerir. Kaydetme/yükleme:

```python
from cryptopredict.models import load_model, save_model

model = get_model("ridge").fit(X, y)
save_model(model, "models/btc_ridge.joblib", horizon=horizon, extra={"symbol": "BTCUSDT"})
model, metadata = load_model("models/btc_ridge.joblib")
```

> `load_model` joblib/pickle kullanır: yalnızca güvendiğiniz dosyaları yükleyin.

## Testler

```bash
pytest -q
```

Testler ağ erişimi gerektirmez; `tests/fixtures/btcusdt_1h_sample.csv` (600
kapanmış BTCUSDT 1h mumu) ve sahte (mock) HTTP yanıtları kullanır.

## Sınırlamalar

- Yalnızca fiyat/hacim tabanlı teknik özellikler; haber, on-chain, emir defteri
  veya duyarlılık verisi yok.
- Klasik modeller (doğrusal ve ağaç tabanlı); derin öğrenme yok. Kısa vadeli
  kripto getirileri gürültüye çok yakındır — modellerin `zero` baseline'ını
  anlamlı biçimde yenememesi **beklenen** bir sonuçtur.
- Nokta tahmini üretilir; güven aralığı / olasılık dağılımı yok.
- Backtest basitleştirilmiştir: kapanış fiyatından anında dolum, sabit
  komisyon/kayma, likidite, fonlama (funding) ve vergi yok; `h > 1` modelleri
  backtest'te yine bir sonraki barın getirisiyle değerlendirilir.
- Tek sembol, tek zaman aralığı; portföy veya çoklu varlık yok.
- Model seçimi aynı walk-forward sonuçlarına bakılarak yapılırsa iyimser yanlılık
  (overfitting) oluşur; son karar için ayrı, hiç görülmemiş bir dönem kullanın.

> ⚠️ **Tekrar: YATIRIM TAVSİYESİ DEĞİLDİR.** Gerçek parayla işlem yapmadan önce
> riskleri anlayın ve gerekirse lisanslı bir danışmana başvurun.

## English summary

**cryptopredict** forecasts the h-step-ahead **log return** of a crypto asset
(`target[t] = log(close[t+h] / close[t])`) from Binance OHLCV candles. It builds
leakage-free technical features (`FeatureConfig`), fits baselines (`zero`, `mean`,
`last`, `ma`) and learned models (`ridge`, `gbm`), and evaluates them with
time-ordered **walk-forward validation** (fresh model per fold, `gap = horizon`)
and a fee/slippage-aware long/flat(/short) **backtest** against buy-and-hold.
Key metrics: MAE, RMSE, sMAPE, directional accuracy and `relative_mae`
(< 1 beats the zero-return baseline); backtest Sharpe, max drawdown and costs.
Install with `pip install -e ".[dev]"` (Python ≥ 3.11), use the `cryptopredict`
CLI (`fetch`, `train`, `predict`, `backtest`; `--csv` for offline use,
`CRYPTOPREDICT_BINANCE_URL` for a mirror if Binance returns HTTP 451), and run
the offline test suite with `pytest`. **Not financial advice** — for education
and research only.
