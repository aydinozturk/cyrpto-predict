# Gerçek veriyle model karşılaştırması

> **Uyarı:** Bu çalışma yazılımın teknik değerlendirmesidir; yatırım tavsiyesi
> değildir. Geçmiş performans gelecekteki sonuçları garanti etmez.

## Kısa sonuç

Hiçbir model, hiçbir sembol/interval/ufuk kesitinde sıfır-getiri (`zero`)
baseline'ını istatistiksel olarak anlamlı biçimde yenmedi. Holm ailesindeki 124
model-kesit karşılaştırmasında Holm düzeltmesinden sonra model lehine anlamlı
sonuç **0**, baseline lehine anlamlı sonuç **98**. Ham `p < 0,05` ile model
lehine tek bir sonuç var: BTCUSDT 1h h=1 `lgbm_cls`, relative MAE 0,9987,
`p=0,039`. Bu sonuç Holm sonrası `p=0,156` oluyor ve tek başına bir kanıt
sayılmamalı. Kareli kayıpta (MSE) ise aynı model `zero`'dan anlamlı biçimde
**kötü** (`p=0,020`); yani bu tek işaret kayıp fonksiyonu seçimine bağlı.

10 bps maliyetli long/flat backtest daha da açık: eşiksiz işlem yapan 94
model-kesitin medyan toplam getirisi %-72,9; buy-and-hold'u geçen yalnız 1 tane
var. Sinyal eşiğini yükseltmek kaybı esas olarak işlem sayısını azaltarak
küçültüyor. Bu, tahmin başarısından değil, daha az maliyet ödenmesinden geliyor.
LSTM/GRU, zaman bütçesinin bağlayıcı olmadığı iki BTC kesitinde de baseline'dan
iyi değil.

## Yeniden üretme

Tablolar commit `ad7221da698149e7eb84e0a08451faf38ba60221` üzerinde, 9 Ekim
2026 UTC'de (run `20261009T075014Z`) üretildi. Bu raporu ekleyen commit yalnız
`docs/results.md`'yi değiştirir; kod aynıdır.

```bash
python scripts/benchmark.py --help
python scripts/benchmark.py --quick --output data/benchmark/quick.csv   # hızlı duman testi
python scripts/benchmark.py --output data/benchmark/final.csv           # bu rapordaki tam koşu
```

Tam koşu varsayılan argümanlarla çalışır: BTCUSDT ve ETHUSDT, 1h/4h/1d, 5
walk-forward kat, 10 bps ücret, mutlak kayıp, Bartlett HAC, seed 42, deep için fold
başına 60 sn/100 epoch. Komut şu dosyaları üretir:

- `final.csv`: model başına metrikler, DM testi, Holm p'si, backtest,
  `deep_epochs`/`deep_time_stops`, süre ve `ok/skipped/error` durumu;
- `final_thresholds.csv`: sıfır dışındaki her başarılı model için 0/2,5/5/10 bps
  sinyal eşiği backtest'i;
- `final_metadata.json`: tam komut, commit, argümanlar, paket sürümleri, Holm
  ailesi tanımı;
- `final.md`: aşağıdaki özet tabloların CSV'den mekanik olarak üretilmiş hali ve
  tüm başarılı satırlar.

Aşağıdaki tablolar `final.md`'deki tabloların Türkçe sayı biçimine çevrilmiş
halidir. Mumlar Binance'in anahtarsız public kline uç noktasından alınıp
`data/benchmark/cache/` altında önbelleklenir. `data/` `.gitignore`
kapsamındadır; ham veri, çıktı ve model dosyaları repoya eklenmez. Koşu CPU'da
yaklaşık 8 dakika sürdü (model süreleri toplamı 479 sn). Ortam: Python 3.11.2,
NumPy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, SciPy 1.17.1, LightGBM 4.7.0,
PyTorch 2.14.1+cpu.

| Sembol | Interval | Veri (UTC) | Mum | Test dönemi (UTC) | OOS gözlem |
|---|---:|---|---:|---|---:|
| BTCUSDT, ETHUSDT | 1h | 2025-01-01 → 2026-10-01 | 15.313 | 2025-04-29 → 2026-09-30 | 12.420-12.460 |
| BTCUSDT, ETHUSDT | 4h | 2022-01-01 → 2026-10-01 | 10.405 | 2022-10-29 → 2026-09-30 | 8.595 |
| BTCUSDT, ETHUSDT | 1d | 2022-01-01 → 2026-10-01 | 1.735 | 2022-12-21 → 2026-09-30 | 1.380 |

## Deney düzeni

- Hedef: `h` bar sonrasına log-getiri. 1h'de `h` = 1/4/24, 4h ve 1d'de `h` = 1.
- Beş zaman sıralı, genişleyen walk-forward kat; dış katlar arasında
  `gap=horizon`. `stack` iç katlarında da `gap=horizon` kullanılır.
- Modeller: `zero`, `mean`, `last`, `ma`, `ridge`, `gbm`, `lgbm`, `lgbm_cls`,
  `ensemble`, `stack`, `lstm`, `gru`. Hepsi sabit varsayılan
  hiperparametrelerle çalıştı. Tuning yapılmadığı için nested tuning de
  gerekmedi. Test dönemine bakılarak hiçbir seçim yapılmadı.
- `--seed 42` her stokastik modele geçer: `gbm`, `lgbm`, `lgbm_cls`,
  `lstm`, `gru` ve `ensemble`/`stack` içindeki `gbm`/`lgbm` üyeleri.
- Özellikler: "zengin" varsayılan tam settir. Ablation için "legacy" set yalnız
  eski getiri, ortalama, SMA/EMA, RSI, MACD, Bollinger, volatilite ve
  hacim-z-score sütunlarıdır; `ridge`, `gbm`, `lgbm` ile koşuldu.
- LSTM/GRU yalnız BTCUSDT 1h h=1 ve 4h h=1'de koşuldu, fold başına en çok 60 sn
  ve 100 epoch. Diğer 16 deep satırı CSV'de `skipped` olarak duruyor.
- Toplam 150 satır: 134 `ok`, 16 `skipped`, 0 `error`.
- Backtest: long/flat, pozisyon değişimi başına 10 bps ücret, slippage 0.
  Pozisyon her ufukta bir bar tutulur ve bir sonraki barın getirisini kazanır.
  Yani h=4/24 sonuçları "h bar elde tut" stratejisi değildir.

## İstatistiksel yöntem

Diebold-Mariano testi MAE ile tutarlı mutlak kayıp farkı üzerinden yapıldı:

`d[t] = |y[t] - model[t]| - |y[t] - 0|`

Negatif DM istatistiği model lehinedir. Ufuk `h` için `h-1` gecikmeli Bartlett
(Newey-West) HAC varyansı, Harvey-Leybourne-Newbold küçük örnek düzeltmesi ve
`t(n-1)` dağılımından iki yönlü p-değeri kullanıldı. HLN düzeltmesi uniform
kernel için türetilmiştir; Bartlett ile kullanımı yaygın bir yaklaşıklıktır.
Test ölçekten bağımsızdır: iki hata serisini aynı sabitle çarpmak istatistiği
değiştirmez. Tüm modeller aynı OOS timestamp'lerinde birebir eşleştirildi.

**Holm ailesi:** aynı sembol, interval, ufuk, özellik seti ve kayıp
fonksiyonunda koşan, `zero` dışındaki tüm modeller. `zero`'nun kayıp farkı
özdeş olarak 0 (p=1) olduğu için aileden çıkarıldı. Toplam 20 aile ve 124 test
var (zengin: 94, legacy: 30).

## Her kesitte en düşük MAE'li model (zengin özellikler)

Bu tablo betimseldir: modeller arasından minimumu seçmek de bir seçimdir.
Anlamlılık için Holm p'sine bakılmalı. Getiri ve B&H toplam getiridir.

| Sembol | Interval | h | Model | MAE | relative_mae | Yön | DM stat | p | Holm p | Getiri | B&H | İşlem |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTCUSDT | 1h | 1 | lgbm_cls | 0,002854 | 0,998733 | %51,38 | -2,064 | 0,039 | 0,156 | %-98,63 | %-12,03 | 4.195 |
| BTCUSDT | 1h | 4 | mean | 0,005795 | 1,000633 | %49,15 | 2,215 | 0,027 | 0,054 | %-27,73 | %-11,14 | 2 |
| BTCUSDT | 1h | 24 | ma | 0,015188 | 1,002126 | %49,94 | 1,383 | 0,167 | 0,167 | %-66,10 | %-10,39 | 1.205 |
| BTCUSDT | 4h | 1 | stack | 0,006289 | 0,999982 | %50,13 | -0,065 | 0,948 | 1,000 | %-63,36 | %299,89 | 2.481 |
| BTCUSDT | 1d | 1 | stack | 0,016772 | 1,000013 | %20,14 | 0,020 | 0,984 | 0,984 | %-14,37 | %403,99 | 123 |
| ETHUSDT | 1h | 1 | lgbm_cls | 0,004268 | 1,000227 | %50,84 | 0,588 | 0,557 | 0,557 | %-90,00 | %48,30 | 2.287 |
| ETHUSDT | 1h | 4 | stack | 0,008850 | 1,000897 | %19,36 | 2,212 | 0,027 | 0,027 | %-45,53 | %49,59 | 474 |
| ETHUSDT | 1h | 24 | ma | 0,023842 | 1,001202 | %48,65 | 0,744 | 0,457 | 0,457 | %-60,63 | %53,37 | 1.230 |
| ETHUSDT | 4h | 1 | lgbm_cls | 0,008400 | 0,999838 | %51,45 | -0,212 | 0,832 | 1,000 | %-91,92 | %64,99 | 2.758 |
| ETHUSDT | 1d | 1 | stack | 0,022474 | 1,001067 | %39,13 | 0,729 | 0,466 | 0,543 | %-27,97 | %122,75 | 194 |

On kesitin yedisinde en iyi sıfır-dışı model bile `zero`'dan kötü
(`relative_mae > 1`). ETHUSDT 1h h=4'te en iyi model (`stack`) Holm sonrası da
anlamlı biçimde kötü. Zengin sette `relative_mae < 1` olan dört satır var
(yukarıdaki üç satır ve ETHUSDT 4h `stack`, 0,999956). Bunların üçünde fark
%0,02'nin altında ve p > 0,8. `stack`'in bazı kesitlerde yön isabetinin düşük
olması (%19-39), negatif olmayan meta ağırlıkların bazı fold'larda tümüyle 0'a
inip tahmini tam 0 yapmasından kaynaklanır. Sıfır tahminler yön metriğinde
isabet sayılmaz.

## DM/Holm özeti

| | Zengin | Legacy | Toplam |
|---|---:|---:|---:|
| Test (model × kesit) | 94 | 30 | 124 |
| Negatif DM istatistiği (model lehine yön) | 4 | 0 | 4 |
| Ham p < 0,05, model lehine | 1 | 0 | 1 |
| Aile içi Holm p < 0,05, model lehine | 0 | 0 | 0 |
| Aile içi Holm p < 0,05, baseline lehine | 70 | 28 | 98 |

Holm, hata oranını 20 ailenin her birinin içinde kontrol eder; 124 testin
tamamı üzerinde global bir kontrol değildir. Model lehine 0 sonuç, global bir
düzeltmede de değişmez. Aile bazlı ayrıntı (`models`, `raw_better`,
`holm_better`, `holm_worse`, `min_p_better`) `final.md`'deki "DM test against
zero per Holm family" tablosundadır.

Tek ham anlamlı sonuç (BTC 1h h=1 `lgbm_cls`) için iki duyarlılık kontrolü
yapıldı:

- **HAC lag:** Aynı OOS tahminlerde `diebold_mariano(..., max_lag=L)` ile
  Newey-West otomatik lag (L=11) p=0,032, L=24 p=0,025 veriyor.
  Aile içi Holm sonrası (bu p ailede 8. sırada, çarpan 4) yine ≥ 0,10. Sonuç
  lag seçimine bağlı değil.
- **Kayıp fonksiyonu:** Kareli kayıpla (`--loss squared`) aynı model `zero`'dan
  anlamlı biçimde **kötü** (stat +2,32, p=0,020). BTCUSDT 4h (+2,22, p=0,026) ve
  ETHUSDT 4h (+2,10, p=0,036) `lgbm_cls` için de durum aynı. Olası açıklama
  (ayrıca test edilmedi): model işareti tutturduğu çok sayıdaki küçük harekette
  mutlak hatayı biraz düşürüyor, ama kaçırdığı büyük hareketlerde kare hatayı
  daha çok artırıyor.

Komut: `python scripts/benchmark.py --symbols BTCUSDT --intervals 1h --horizons 1
--models zero lgbm_cls --loss squared --no-ablation --output data/benchmark/sq.csv`
(4h için `--symbols BTCUSDT ETHUSDT --intervals 4h`). Bu koşuların Holm aileleri
yalnız `lgbm_cls`'den oluştuğu için burada ham p verildi.

`benchmark.py`'de lag argümanı yok. Lag değerleri, repo kökünde (tam koşu
`data/benchmark/cache/`'i doldurduktan sonra) şu snippet'le üretilir. Model
parametreleri ve seed `benchmark.py` varsayılanlarından alınır:

```python
import sys
sys.path.insert(0, "scripts")
import benchmark as bm
from cryptopredict.evaluation.stats import diebold_mariano
from cryptopredict.features import FeatureConfig, make_dataset
from cryptopredict.pipeline import load_ohlcv, run_backtest

args = bm.parse_args([])
df = load_ohlcv(symbol="BTCUSDT", interval="1h", start=args.start_1h, end=args.end, cache_dir=args.cache_dir)
report = run_backtest(
    df, "lgbm_cls", horizon=1, n_splits=args.splits, fee_bps=args.fee_bps, interval="1h",
    feature_config=FeatureConfig(), dataset=make_dataset(df, horizon=1, config=FeatureConfig()),
    model_params=bm._model_params("lgbm_cls", args),
)
p = report.evaluation.predictions
y, e = p["y_true"].to_numpy(), (p["y_true"] - p["y_pred"]).to_numpy()
nw = int(4 * (len(y) / 100) ** (2 / 9))  # Newey-West otomatik lag: 11
for lag in (0, nw, 24):
    r = diebold_mariano(e, y, horizon=1, loss="absolute", max_lag=lag)
    print(f"lag {lag}: stat={r.statistic:+.3f} p={r.p_value:.3f}")
# lag 0: -2.064 p=0.039 | lag 11: -2.146 p=0.032 | lag 24: -2.243 p=0.025
```

## Model özeti (zengin özellikler, 10 kesit)

| Model | Koşul | Ort. relative_mae | En iyi | `<1` | Holm daha iyi | Holm daha kötü | Medyan getiri |
|---|---:|---:|---:|---:|---:|---:|---:|
| stack | 10 | 1,002377 | 0,999956 | 2 | 0 | 5 | %-69,9 |
| lstm | 2 | 1,002772 | 1,002228 | 0 | 0 | 1 | %-87,1 |
| mean | 10 | 1,003079 | 1,000102 | 0 | 0 | 4 | %-30,8 |
| gru | 2 | 1,004696 | 1,003779 | 0 | 0 | 1 | %-70,9 |
| lgbm_cls | 10 | 1,005453 | 0,998733 | 2 | 0 | 3 | %-76,4 |
| lgbm | 10 | 1,009669 | 1,000166 | 0 | 0 | 8 | %-52,3 |
| ma | 10 | 1,025353 | 1,001202 | 0 | 0 | 8 | %-60,8 |
| ensemble | 10 | 1,041055 | 1,005343 | 0 | 0 | 10 | %-78,3 |
| ridge | 10 | 1,072346 | 1,016419 | 0 | 0 | 10 | %-80,3 |
| gbm | 10 | 1,122116 | 1,030117 | 0 | 0 | 10 | %-84,3 |
| last | 10 | 1,329453 | 1,025076 | 0 | 0 | 10 | %-99,7 |

Deep modeller yalnız iki kesitte koşulduğu için ortalamaları diğer satırlarla
doğrudan karşılaştırılmamalı.

## LSTM/GRU (zaman bütçeli)

| Veri | Model | relative_mae | Yön | DM stat | p | Holm p | Getiri | İşlem | Epoch/fold | Zaman durdurma | Süre |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| BTCUSDT 1h h1 | LSTM | 1,003315 | %51,15 | 2,606 | 0,009 | 0,046 | %-94,61 | 2.607 | 13/11/14/12/13 | 0/5 | 43,9 sn |
| BTCUSDT 1h h1 | GRU | 1,005614 | %51,40 | 3,490 | 4,8e-04 | 0,003 | %-94,41 | 2.747 | 13/15/14/17/18 | 0/5 | 127,6 sn |
| BTCUSDT 4h h1 | LSTM | 1,002228 | %51,04 | 1,471 | 0,141 | 0,424 | %-79,59 | 1.877 | 12/11/11/13/11 | 0/5 | 20,3 sn |
| BTCUSDT 4h h1 | GRU | 1,003779 | %52,10 | 2,083 | 0,037 | 0,149 | %-47,32 | 2.345 | 12/13/17/11/11 | 0/5 | 68,7 sn |

Hiçbir fold'da 60 sn sınırına ulaşılmadı (`deep_time_stops` = 0/5). Eğitim her
fold'da 11-18 epoch'ta, validasyon kaybına göre erken durdurmayla (patience 10)
bitti. Yani bu sonuçlar
kesilmiş bir eğitimden gelmiyor; bütçe karşılaştırmayı haksızlaştırmadı. Süre
tabanlı durdurma donanıma bağlıdır. Daha yavaş bir makinede sınır bağlayıcı olursa
sonuçlar değişebilir; `deep_epochs` sütunu bunu denetlemek için var. BTC 1h'de
LSTM ve GRU baseline'dan anlamlı biçimde **kötü**. 4h'de fark anlamlı değil.
Bu sonuçlar yalnız test edilen varsayılan mimari ve iki kesit için geçerli;
"derin modeller genelde başarısız" diye genellenemez.

## Zengin özellik ablation'ı

`ridge`, `gbm` ve `lgbm` aynı hedef ve timestamp'lerde legacy ve zengin
sütunlarla koşuldu (pozitif fark = zengin set daha kötü).

| Model | Koşul | Zengin daha iyi | Legacy ort. | Zengin ort. | Fark |
|---|---:|---:|---:|---:|---:|
| ridge | 10 | 0 | 1,026813 | 1,072346 | +0,045533 |
| gbm | 10 | 1 | 1,090604 | 1,122116 | +0,031512 |
| lgbm | 10 | 3 | 1,005927 | 1,009669 | +0,003742 |

30 eşleşmenin yalnız dördünde zengin set relative MAE'yi düşürdü. En büyük
kötüleşme ETHUSDT 1d `ridge`'de (+0,1718), en büyük iyileşme ETHUSDT 1d `gbm`'de
(-0,0044). Ek sütunlar, düzenlileştirme veya özellik seçimi ayarlanmadan
varsayılan modellerde sinyale dönüşmüyor; daha çok varyans ekliyor.

## Sinyal eşiği duyarlılığı

Eşikler aynı OOS tahminlere **sonradan** uygulandı. Test döneminde hiçbir eşik
veya model seçilmedi. Bu yüzden aşağıdaki sayılar ayarlanmış bir kuralın elde
edilebilir getirisi değil; maliyet duyarlılığının betimlemesi.

Zengin özelliklerde sıfır dışındaki her başarılı model (94 model-kesit):

| Eşik | Model-kesit | Medyan getiri | Pozitif | B&H'ı geçen | Medyan işlem |
|---:|---:|---:|---:|---:|---:|
| 0 bps | 94 | %-72,9 | 7 | 1 | 1.229 |
| 2,5 bps | 94 | %-56,4 | 8 | 4 | 941 |
| 5 bps | 94 | %-48,4 | 10 | 7 | 741 |
| 10 bps | 94 | %-31,8 | 9 | 8 | 454 |

Eşik yükseldikçe medyan kayıp küçülüyor, çünkü işlem sayısı düşüyor. B&H'ı geçen
20 model/eşik satırının 16'sı BTC 1h'nin negatif B&H döneminde (%-10 ile %-12)
az işlem yapan ya da hiç işlem yapmayan modeller. Flat kalmak orada mekanik
olarak B&H'ı geçer; bu bir tahmin başarısı değil. Pozitif B&H dönemlerinde B&H'ı
geçen yalnız dört satır var: ETHUSDT 1d `ma` (0/2,5/5 bps) ve ETHUSDT 4h `lgbm`
(10 bps, %70,2'ye karşı %65,0). 376 denemede bu kadarı tesadüfle uyumlu.

Önceden sabitlenmiş iki model (`ridge` ve `lgbm`, `--threshold-focus`
varsayılanı) için kesit bazında:

| Veri | Model | 0 bps | 2,5 bps | 5 bps | 10 bps | İşlem 0→10 bps | B&H |
|---|---|---:|---:|---:|---:|---:|---:|
| BTC 1h h1 | lgbm | %-72,0 | %-17,4 | %-4,8 | %0,0 | 1.054→0 | %-12,0 |
| BTC 1h h1 | ridge | %-97,3 | %-94,0 | %-81,2 | %-39,6 | 3.679→656 | %-12,0 |
| BTC 1h h4 | lgbm | %-39,6 | %-10,4 | %-11,0 | %-12,3 | 492→208 | %-11,1 |
| BTC 1h h4 | ridge | %-87,8 | %-86,9 | %-82,0 | %-74,8 | 1.974→1.172 | %-11,1 |
| BTC 1h h24 | lgbm | %-27,0 | %-26,1 | %-28,5 | %-25,2 | 172→106 | %-10,4 |
| BTC 1h h24 | ridge | %-73,8 | %-73,3 | %-73,8 | %-70,9 | 1.220→1.100 | %-10,4 |
| BTC 4h h1 | lgbm | %-62,9 | %-39,1 | %-32,8 | %-26,1 | 1.835→528 | %299,9 |
| BTC 4h h1 | ridge | %-67,2 | %-75,9 | %-85,6 | %-56,6 | 2.337→1.557 | %299,9 |
| BTC 1d h1 | lgbm | %-48,1 | %-61,9 | %-54,4 | %-54,2 | 339→344 | %404,0 |
| BTC 1d h1 | ridge | %45,7 | %23,9 | %6,0 | %-7,6 | 329→353 | %404,0 |
| ETH 1h h1 | lgbm | %-81,9 | %-35,7 | %-21,5 | %-15,7 | 1.326→52 | %48,3 |
| ETH 1h h1 | ridge | %-95,6 | %-90,6 | %-83,8 | %-58,5 | 3.385→1.010 | %48,3 |
| ETH 1h h4 | lgbm | %-48,6 | %-37,4 | %-19,7 | %-6,9 | 284→28 | %49,6 |
| ETH 1h h4 | ridge | %-87,7 | %-83,5 | %-83,2 | %-75,0 | 1.938→1.460 | %49,6 |
| ETH 1h h24 | lgbm | %-53,6 | %-51,5 | %-50,9 | %-27,9 | 172→270 | %53,4 |
| ETH 1h h24 | ridge | %-74,3 | %-76,6 | %-78,5 | %-74,7 | 1.228→1.148 | %53,4 |
| ETH 4h h1 | lgbm | %-76,0 | %-46,1 | %25,3 | %70,2 | 1.827→246 | %65,0 |
| ETH 4h h1 | ridge | %-86,4 | %-85,2 | %-85,8 | %-87,6 | 2.316→1.532 | %65,0 |
| ETH 1d h1 | lgbm | %-51,0 | %-56,2 | %-31,3 | %-37,2 | 239→70 | %122,8 |
| ETH 1d h1 | ridge | %-35,1 | %-37,6 | %-53,8 | %-61,8 | 343→321 | %122,8 |

1d ve h=24'te tahminler eşiğin etrafında dolaştığı için eşik yükselince işlem
sayısı artabiliyor. Tutarlı bir "doğru eşik" yok. Bir eşik seçilecekse yalnız
eğitim/validasyon pencerelerinde (nested walk-forward) seçilmeli ve ayrı bir son
test döneminde doğrulanmalı.

## Önceki raporla karşılaştırma

Önceki rapor 2026-01-01→2026-10-01 saatlik veriyi, 5 katı ve o zamanki özellik
setini kullanıyordu. Yeni koşu 2025 başından başlıyor ve zengin özellik setini
kullanıyor. Bu yüzden fark kontrollü bir ablation değil. 1h h=1 relative MAE:

| Model | BTC önceki | BTC yeni | ETH önceki | ETH yeni |
|---|---:|---:|---:|---:|
| zero | 1,0000 | 1,0000 | 1,0000 | 1,0000 |
| mean | 1,0013 | 1,0001 | 1,0011 | 1,0009 |
| ridge | 1,0109 | 1,0164 | 1,0128 | 1,0227 |
| ma | 1,0367 | 1,0364 | 1,0352 | 1,0388 |
| gbm | 1,0634 | 1,0301 | 1,0598 | 1,0469 |
| last | 1,4789 | 1,4624 | 1,4937 | 1,4684 |

Ana sonuç değişmedi: klasik modeller `zero`'yu yenmiyor. Yeni modeller
(`lgbm`, `lgbm_cls`, `ensemble`, `stack`, `lstm`, `gru`) de bu tabloyu
değiştirmedi. `gbm`'deki iyileşme daha uzun eğitim verisinden geliyor olabilir.
`ridge` zengin sette daha kötü; ablation sonucuyla tutarlı.

## Sonuç ve sınırlar

- **Hangi model/ufuk/interval anlamlı iyileşme veriyor?** Hiçbiri. Holm sonrası
  model lehine anlamlı sonuç yok. Ham p < 0,05 olan tek sonuç (BTC 1h h=1
  `lgbm_cls`) düzeltme sonrası anlamsız.
- **Hangisi vermiyor?** Geri kalanı. 124 testin 98'inde model baseline'dan
  anlamlı biçimde kötü. `ridge`, `gbm`, `ensemble` ve `last` her kesitte kötü.
- Daha uzun ufuk (h=4/24) ve daha düşük frekans (4h/1d) tek başına sinyal
  üretmedi.
- Zengin özellik seti varsayılan modellerde çoğunlukla performansı düşürdü.
- 10 bps maliyetle neredeyse tüm stratejiler sermayenin büyük kısmını kaybediyor.
  Eşik duyarlılığı post-hoc ve betimsel.
- Deep sonuçları yalnız iki BTC kesiti, varsayılan mimari ve bu CPU için
  geçerli.
- `zero`, modellerin iç içe (nested) olduğu özel bir durum. Genişleyen pencerede
  DM bu durumda muhafazakâr olabilir (Clark ve McCracken 2001; Clark ve West
  2007, kareli kayıp için düzeltilmiş test). Giacomini ve White (2006)
  çerçevesi tahmin edilmiş parametreli ve iç içe modellerde DM tipi testi
  geçerli kılar, ama sabit boylu kayan (rolling) pencere ister; burada pencere
  genişleyen. Dolayısıyla "anlamsız" sonuçların bir kısmı düşük güçten
  gelebilir. Ancak farklar zaten
  ekonomik olarak çok küçük (relative MAE farkı en çok %0,13).
- Backtest spread, slippage, fonlama, likidite ve vergi içermez. Gerçek
  uygulama daha kötü olur.

Bu verilerle dürüst cevap şu: "hangi model sıfır-getiri tahminini anlamlı
biçimde geçiyor?" sorusunun cevabı **hiçbiri**, "hangi koşul ekonomik olarak
ikna edici?" sorusunun cevabı da **hiçbiri**.

**Bu rapor ve yazılım yatırım tavsiyesi değildir.**
