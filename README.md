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
- [Dashboard & Docker](#dashboard--docker)
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
Opsiyonel gruplar (extras):

| Extra | Ne ekler | Kurulum |
|---|---|---|
| `ml` | LightGBM: `lgbm`, `lgbm_cls`, `ensemble`/`stack`'in varsayılan üyesi | `pip install -e ".[dev,ml]"` |
| `web` | FastAPI paneli (`cryptopredict-web`) | `pip install -e ".[dev,web,ml]"` |
| `deep` | PyTorch: `lstm`, `gru` (ağır) | önce CPU torch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`, sonra `pip install -e ".[dev,ml,deep]"` |

Bağımlılığı kurulu olmayan model `available_models()` listesinde ve CLI
seçeneklerinde görünmez.
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

# 5) Hiperparametre araması (nested walk-forward), sonra önerilen parametrelerle eğit
cryptopredict tune --symbol BTCUSDT --interval 1h --start 2024-01-01 \
    --model lgbm --n-trials 20 --time-budget 60   # saniye, arama başına (5 kat + final)
cryptopredict train --symbol BTCUSDT --interval 1h --start 2024-01-01 \
    --model lgbm --param n_estimators=300 --param learning_rate=0.03 --out models/btc_lgbm.joblib
```

Ağ erişimi olmadan denemek için her komut `--csv PATH` ile yerel bir OHLCV
dosyası kabul eder; repodaki örnek veri (600 saatlik BTCUSDT mumu):

```bash
CSV=tests/fixtures/btcusdt_1h_sample.csv
cryptopredict backtest --csv $CSV --model all --splits 5
cryptopredict train    --csv $CSV --model ridge --horizon 1 --out models/x.joblib
cryptopredict predict  --csv $CSV --model-path models/x.joblib --json
cryptopredict tune     --csv $CSV --model ridge --splits 3 --n-trials 5
```

`predict`, aralık biliniyorsa (`--interval` ya da modelin metadata'sı) henüz
kapanmamış son mumları (`open_time + aralık > şimdi`) atar ve stderr'e not düşer;
`--include-open-bar` bu kontrolü kapatır. Aralık bilinmiyorsa `--csv`'deki tüm
satırlar kapanmış kabul edilir.

### Komutlar ve bayraklar

**Ortak veri bayrakları** (`fetch`, `train`, `predict`, `backtest`, `tune`):

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
| `train` | `--model NAME --horizon H --out PATH.joblib [--param K=V ...]` | Tüm veriyle modeli eğitir; model, parametreleri (`model_params`), özellik yapılandırması, sembol/aralık ve eğitim dönemiyle birlikte kaydeder (varsayılan model `ridge`, horizon `1`) |
| `predict` | `--model-path PATH [--csv PATH] [--include-open-bar]` | Son **kapanmış** mum için tahmini log-getiri, beklenen fiyat, yön (yukarı/aşağı) ve horizon'u basar. `--csv` yoksa modelin metadata'sındaki sembol/aralık için yeterli geçmiş çekilir: özellik ısınması (`required_history`) + modelin `lookback`'i + 50 bar pay, en az 500 bar (CSV ile eğitilmiş modelde `--symbol` verin) |
| `backtest` | `--model NAME\|all --horizon H --splits K --fee-bps F [--slippage-bps S] [--threshold T] [--da-threshold D] [--allow-short] [--include-heavy] [--param K=V ...]` | Walk-forward kat metrikleri, genel metrikler ve strateji vs. buy-and-hold; `all` ile modelleri RMSE'ye göre sıralı bir tabloda karşılaştırır (ağır modeller yalnız `--include-heavy` ile). Varsayılanlar: `--splits 5`, `--fee-bps 10`, `gap = horizon`. `--threshold` yalnız pozisyon eşiğidir (`\|tahmin\| > T` ise işlem); `--da-threshold` yön isabetindeki nötr banttır (`\|gerçek\| <= D` olan barlar hariç). İkisinin de varsayılanı 0. `--param` yalnız tek modelle kullanılır |
| `tune` | `--model NAME --horizon H --splits K [--inner-splits I] [--n-trials N] [--time-budget SN] [--metric mae] [--seed 42] [--param K=V ...] [--space K=[...] ...]` | Nested walk-forward hiperparametre araması: her dış kat yalnız kendi eğitim satırlarında aranır, test katı bir kez skorlanır (dürüst tahmin). Sonra aynı arama tüm veride koşulur ve `train` için `--param` önerisi basılır. Bkz. [Parametreler ve tuning](#parametreler-ve-tuning) |

Model adları: `zero`, `mean`, `last`, `ma`, `ridge`, `gbm`; `ml` extra'sıyla
`lgbm`, `lgbm_cls`, `ensemble`, `stack`; `deep` extra'sıyla `lstm`, `gru`
(bkz. [Modeller](#modeller)).

`predict --json` çıktısının anahtarları: `model`, `symbol`, `interval`, `horizon`,
`as_of` (son kapanmış mumun `open_time`'ı), `target_time` (tahmin edilen
kapanışın zamanı: `as_of + (horizon + 1) × aralık`), `last_close`,
`predicted_log_return`, `predicted_pct_change` (yüzde), `expected_price`
(`last_close * exp(predicted_log_return)`), `direction` (`up` / `down` / `flat`) ve
`dropped_open_bars` (atlanan kapanmamış mum sayısı).

CSV biçimi: `open_time` (UTC zaman damgası) ve `open,high,low,close,volume` kolonları.

## Dashboard & Docker

FastAPI tabanlı yönetim paneli; veri çekme, model eğitme, kayıtlı modellerden
tahmin üretme, walk-forward backtest çalıştırma ve arka plan işlerini izleme
ekranlarını tek serviste sunar. Yerel imajı oluşturup paneli yalnızca bu makinede
açmak için:

```bash
cp .env.example .env
docker compose up -d --build dashboard
```

İmaj varsayılan olarak `web,ml` extra'larıyla (LightGBM dahil) kurulur. LSTM/GRU
için CPU PyTorch'lu (çok daha büyük) bir imaj:

```bash
docker build --build-arg EXTRAS=web,ml,deep -t cryptopredict:deep .
# ya da .env'de: CRYPTOPREDICT_EXTRAS=web,ml,deep ve CRYPTOPREDICT_IMAGE=cryptopredict:deep
```

Panel `http://127.0.0.1:8000` adresindedir. Portu ağdaki başka makinelere
açacaksanız `.env` içinde güçlü bir `CRYPTOPREDICT_DASHBOARD_TOKEN` tanımlayın.
GHCR/Docker Hub yayın ayarları, private GHCR girişi, volume yönetimi, güncelleme
komutları ve uçtan uca panel testi için [Dashboard ve Docker dağıtım
rehberine](docs/deploy.md) bakın.

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
hata fırlatır, `"ignore"` sessiz geçer. Uyarıyla devam edilen boşluklu veri
güvenlidir: özellik katmanı boşlukları aşan pencere ve hedefleri veri setinden
çıkarır (bkz. [Mum boşlukları](#mum-boşlukları)).

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
| `bollinger_pct_b_20`, `bollinger_width_20` | Bollinger bandı (20, 2σ) içindeki konum ve bant genişliği (sabit fiyatlı pencerede `0.5` ve `0`) |
| `volatility_24` | Son 24 barın 1-bar log-getiri standart sapması |
| `volume_zscore_24` | Hacmin 24 barlık z-skoru (sabit hacimde `0`) |
| `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` | Saat / haftanın günü (yalnızca `add_time_features=True` ise) |
| `atr_14` | Wilder ATR'nin kapanışa oranı (`ATR / close`) |
| `stoch_k_14`, `stoch_d_14` | Stokastik %K (0–100, düz pencerede `50`) ve 3 barlık ortalaması %D |
| `obv_zscore_24`, `signed_volume_ratio_24` | OBV'nin 24 barlık z-skoru (ham OBV değil) ve işaretli hacmin toplam hacme oranı (−1…1) |
| `parkinson_vol_24`, `garman_klass_vol_24` | High/low (ve open/close) aralığından bar başına volatilite tahmini |
| `candle_body`, `candle_upper_wick`, `candle_lower_wick`, `candle_log_range` | Mum gövdesi (işaretli) ve fitillerin aralığa oranı, `log(high / low)` (`add_candle_features`) |
| `ret_skew_72`, `ret_kurt_72` | Son 72 barın 1-bar log-getiri çarpıklığı ve fazla basıklığı |
| `htf_4h_*`, `htf_1d_*` | Üst zaman dilimi: son **kapanmış** üst mumun `log_ret_1`, `log_ret_3`, `rsi_14` ve `close / EMA14 − 1` değerleri |

Kolon adları pencere parametrelerinden türetilir (ör. `rsi_window=21` → `rsi_21`).
Varsayılanlar `FeatureConfig(...)` ya da kısmi bir sözlükle değiştirilebilir:
`make_dataset(ohlcv, horizon=1, config={"rsi_window": 21, "add_time_features": True})`.
`FeatureConfig.to_dict()` / `FeatureConfig.from_dict()` yapılandırmayı model
metadata'sına yazıp tahminde aynı özellikleri yeniden üretmeye yarar.

Üst zaman dilimi (`htf_intervals=("4h", "1d")`): her segment UTC gece yarısına
hizalı üst mumlara bölünür ve yalnız **tamamı mevcut ve kapanmış** mumlar kullanılır.
`t` barı, kapanışı (`open_time + htf`) kendi kapanışından (`t + bar`) sonra olmayan son
üst mumu görür; oluşmakta olan mum asla sızmaz. Bir üst zaman dilimi yalnız
`bar < htf ≤ htf_max_ratio · bar` (varsayılan 24) ise kullanılır: 1d verisinde hiç,
15m verisinde yalnız 4h. Günlük göstergeler ısınmayı uzatır: 1h veride segment
başına ~16 gün (`features.required_history(config, bar)` üst sınırı verir;
`htf_intervals=()` ile kapatılır).

- `features.build_features(df, config, bar=None)` — tüm özellikler, `df` ile aynı
  indeks; ısınma (warmup) satırları NaN kalır (serinin başında ve her boşluktan sonra).
- `features.make_dataset(df, horizon, config, bar=None)` → `(X, y)`: ısınma
  satırları, hedefi henüz bilinmeyen son `h` satır ve hedefi bir boşluğu aşan
  satırlar atılır; `X` ve `y` aynı indeksi paylaşır, NaN/sonsuz değer içermez
  (impute yapılmaz).
- `features.latest_features(df, config, bar=None, columns=None)` — canlı tahmin için **son barın**
  özellik satırı. O satır eksikse (yetersiz geçmiş ya da son bar bir boşluktan
  hemen sonra ısınmada) daha eski bir satıra düşmek yerine `ValueError` verir.
  `columns` verilirse (ör. modelin eğitildiği kolonlar) yalnız onlar döner ve
  yalnız onların dolu olması gerekir.
- `features.required_history(config, bar)` — son satırın dolu olması için gereken
  kesintisiz bar sayısının üst sınırı (canlı tahminde çekilecek geçmiş için).
- `features.forward_log_return(close, horizon, bar=None)` /
  `features.make_target(df, horizon, bar=None)` — hedef serisi (boşluğu aşan değerler NaN).
- `features.infer_bar(index)` / `features.segment_ids(index, bar=None)` — bar süresi
  ve kesintisiz segment numaraları (aşağıya bakın).

### Mum boşlukları

Borsa bakım/kesintilerinde bazı mumlar hiç yoktur. Boşluktan sonraki bar bitişik
sayılsaydı `h=1` hedefi aslında 2 barlık getiri olur, pencereler kesintinin
üzerinden hesaplanırdı. Bunun yerine:

- Bar süresi indeksteki en sık adımdır (`infer_bar`; eşitlikte en küçüğü) ya da
  `bar=pd.Timedelta("1h")` ile açıkça verilir. Bundan uzun her adım yeni bir
  **segment** başlatır (`segment_ids`; boşluksuz veride hepsi 0).
- Göstergeler her segmentte ayrı hesaplanır: boşluktan sonraki satırlar serinin
  başındaki kadar ısınma NaN'ı alır ve EMA/MACD/RSI durumu boşluğu aşmaz.
- `t` ve `t + h` farklı segmentteyse `target[t]` NaN olur; `make_dataset` bu
  satırları atar.
- Boşluksuz veride çıktı bu davranıştan önceki sürümle birebir aynıdır.

## Modeller

| Ad | Sınıf | Tahmin |
|---|---|---|
| `zero` | `ZeroReturn` | Her zaman 0 (rastgele yürüyüş: "fiyat değişmez") — ana baseline |
| `mean` | `MeanReturn` | Eğitim hedeflerinin ortalaması (sabit drift) |
| `last` | `LastReturn` | Son gözlenen getiri, `log_ret_1` (momentum) |
| `ma` | `MovingAverageReturn` | Son `window` (24) eğitim hedefinin ortalaması; `column="ret_mean_24"` verilirse o kolon (CLI bunu kullanır) |
| `ridge` | `RidgeForecaster` | StandardScaler + Ridge regresyon (`alpha=1.0`) |
| `gbm` | `GBMForecaster` | HistGradientBoostingRegressor (deterministik, `random_state=42`) |
| `lgbm` | `LGBMForecaster` | LightGBM regresyon; eğitim penceresinin son %15'iyle early stopping, sonra tüm pencereye refit (`ml`) |
| `lgbm_cls` | `LGBMDirectionForecaster` | LightGBM yön sınıflandırıcı; olasılık, eğitimdeki koşullu ortalamalarla (`E[y\|up]`, `E[y\|down]`) getiriye çevrilir (`ml`) |
| `ensemble` | `EnsembleForecaster` | Üyelerin (varsayılan `ridge`, `gbm`, `lgbm`) ağırlıklı ortalaması (`ml`) |
| `stack` | `StackingForecaster` | Üye tahminleri üzerinde negatif olmayan Ridge meta-model; eğitim penceresindeki kronolojik out-of-fold tahminlerle eğitilir. CLI/panel `gap`'i horizon'a eşitler (`ml`) |
| `lstm` | `LSTMForecaster` | PyTorch LSTM; son `lookback` (48) özellik satırından tahmin, zaman bütçeli CPU eğitimi (`deep`, **ağır**) |
| `gru` | `GRUForecaster` | `lstm` ile aynı, GRU hücresi (`deep`, **ağır**) |

Tüm modeller scikit-learn `BaseEstimator` tabanlıdır (`sklearn.base.clone`
çalışır). Ölçekleme pipeline'ın içindedir, yani walk-forward'da yalnızca eğitim
penceresine fit edilir. Model `models.get_model("ridge", alpha=10.0)` ile
oluşturulur; `models.available_models()` kurulu bağımlılığı olan adları,
`models.default_compare_models()` ise ağır (`heavy`) olmayanları döner.
`backtest --model all` ve paneldeki `all` yalnız ağır olmayanları karşılaştırır;
`lstm`/`gru` için `--include-heavy` (panelde `include_heavy: true`) gerekir.

### Parametreler ve tuning

`train`, `backtest` ve `tune` için `--param KEY=VALUE` tekrarlanabilir. Değer
JSON olarak okunur (`50`, `0.05`, `true`, `null`, `["ridge","gbm"]`); Python
yazımı `True`/`False`/`None` de olur, gerisi metindir. Bilinmeyen parametre,
veri yüklenmeden hata verir. Eğitilen modelin parametreleri metadata'da
`model_params` olarak saklanır.

```bash
cryptopredict train --csv $CSV --model ridge --param alpha=100 --out models/r.joblib
cryptopredict backtest --csv $CSV --model lgbm --param num_leaves=7 --param min_child_samples=100
cryptopredict train --csv $CSV --model stack --horizon 4 \
    --param 'members=["ridge","gbm","lgbm"]' --out models/stack.joblib   # gap=4 otomatik
cryptopredict tune --csv $CSV --model ridge --space 'alpha=[0.1,1,10,100]' --json
```

`tune` hiperparametreleri sızıntısız seçer: her dış walk-forward katında arama
yalnız o katın eğitim satırlarında, iç walk-forward ile yapılır; dış test katına
seçim sırasında hiç bakılmaz. Raporlanan "overall" metrikleri bu yüzden ayarlanmış
modelin dürüst tahminidir. Ardından aynı arama tüm satırlarda koşulur ve
`best_params` (bunun için skor iddiası yok) `train --param ...` önerisi olarak
basılır. Arama uzayı `--space` ile verilmezse modelin yerleşik uzayı
(`tuning.DEFAULT_SPACES`: `ridge`, `gbm`, `lgbm`, `lgbm_cls`) kullanılır; ilk aday
her zaman varsayılanlardır. `--n-trials` arama başına aday sayısını,
`--time-budget` arama başına saniye bütçesini sınırlar (bütçe dolunca çalışan aday
biter, en iyisi döner). Dış/iç katlar arasında `gap = horizon` kullanılır;
`stack` için de `gap = horizon` geçilir.

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
- Derin modeller (`lstm`, `gru`) yalnız CPU'da ve zaman bütçesiyle eğitilir; büyük
  ağlar ve GPU hedeflenmez. Kısa vadeli kripto getirileri gürültüye çok
  yakındır — modellerin `zero` baseline'ını anlamlı biçimde yenememesi
  **beklenen** bir sonuçtur.
- Nokta tahmini üretilir; güven aralığı / olasılık dağılımı yok.
- Backtest basitleştirilmiştir: kapanış fiyatından anında dolum, sabit
  komisyon/kayma, likidite, fonlama (funding) ve vergi yok; `h > 1` modelleri
  backtest'te yine bir sonraki barın getirisiyle değerlendirilir.
- Tek sembol, tek zaman aralığı; portföy veya çoklu varlık yok.
- Mum boşluklarında atılan satırlar backtest'te de yoktur: boşluk ve sonrasındaki
  ısınma dönemindeki getiriler (strateji ve buy-and-hold için) hesaba girmez.
  Walk-forward katları satır sayısına göre bölünür, yani boşluklu veride katlar
  eşit süreli olmayabilir. Takvim ayı (`1M`) gibi değişken uzunluklu barlar
  desteklenmez.
- Model seçimi aynı walk-forward sonuçlarına bakılarak yapılırsa iyimser yanlılık
  (overfitting) oluşur; son karar için ayrı, hiç görülmemiş bir dönem kullanın.

> ⚠️ **Tekrar: YATIRIM TAVSİYESİ DEĞİLDİR.** Gerçek parayla işlem yapmadan önce
> riskleri anlayın ve gerekirse lisanslı bir danışmana başvurun.

## English summary

**cryptopredict** forecasts the h-step-ahead **log return** of a crypto asset
(`target[t] = log(close[t+h] / close[t])`) from Binance OHLCV candles. It builds
leakage-free technical features (`FeatureConfig`; computed per contiguous
segment so neither features nor targets span missing candles), fits baselines (`zero`, `mean`,
`last`, `ma`) and learned models (`ridge`, `gbm`; with the `ml` extra `lgbm`,
`lgbm_cls`, `ensemble`, `stack`; with the `deep` extra the heavy `lstm`, `gru`),
and evaluates them with
time-ordered **walk-forward validation** (fresh model per fold, `gap = horizon`)
and a fee/slippage-aware long/flat(/short) **backtest** against buy-and-hold.
Key metrics: MAE, RMSE, sMAPE, directional accuracy and `relative_mae`
(< 1 beats the zero-return baseline); backtest Sharpe, max drawdown and costs.
Install with `pip install -e ".[dev]"` (Python ≥ 3.11), use the `cryptopredict`
CLI (`fetch`, `train`, `predict`, `backtest`, `tune`; `--param KEY=VALUE` for
model parameters, `backtest --model all --include-heavy` to also compare heavy
models, `tune` for a nested walk-forward hyperparameter search; `--csv` for offline use,
`CRYPTOPREDICT_BINANCE_URL` for a mirror if Binance returns HTTP 451), and run
the offline test suite with `pytest`. **Not financial advice** — for education
and research only.
