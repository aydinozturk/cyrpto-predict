# Gerçek veriyle uçtan uca model karşılaştırması

> **Uyarı:** Bu çalışma yalnızca yazılımın teknik değerlendirmesidir; yatırım
> tavsiyesi değildir. Geçmiş performans gelecekteki sonuçları garanti etmez.

## Deney düzeni

Deney 8 Ekim 2026 UTC tarihinde, Binance'in anahtarsız public kline uç noktası
(`https://api.binance.com/api/v3/klines`) kullanılarak çalıştırıldı. BTCUSDT ve
ETHUSDT için 1 saatlik 6.553 kapanmış mum alındı. Kullanılan kod tabanı
`origin/main` üzerindeki `b5db5f9` commit'iydi:

> **Kod tabanı ve doğrulama:** Sonuçlar `b5db5f9`'da üretildi, `3065f05`'te
> yeniden doğrulandı. Aynı CSV'ler `--csv` ile `3065f05` üzerinde yeniden
> çalıştırıldı: `backtest --model all --splits 5 --fee-bps 10` (ve ücret
> duyarlılığı için `--fee-bps 0`) ile `train`/`predict` çıktıları bu rapordaki
> tablolarla birebir aynı çıktı. Veride 0 mum boşluğu var (her iki sembolde
> 6.553 ardışık saatlik mum). Bu nedenle #33'teki boşluk düzeltmesi bu sayıları
> değiştirmiyor. #34'ten sonra `predict` kapanmamış son mumu atlıyor; bu
> CSV'lerde son mum kapanmış olduğundan `dropped_open_bars` değeri 0.

| Sembol | Başlangıç (UTC) | Bitiş (UTC) | Mum sayısı |
|---|---:|---:|---:|
| BTCUSDT | 2026-01-01 00:00 | 2026-10-01 00:00 | 6.553 |
| ETHUSDT | 2026-01-01 00:00 | 2026-10-01 00:00 | 6.553 |

Ana değerlendirme ayarları:

- hedef: bir sonraki 1 saatlik log-getiri (`horizon=1`);
- zaman sıralı 5 katlı walk-forward doğrulama ve katlar arasında 1 bar boşluk;
- backtest: long/flat, short kapalı, sinyal eşiği 0, slippage 0;
- her pozisyon değişiminde 10 baz puan ücret;
- toplam 5.430 out-of-sample tahmin/bar.

Çalıştırılan temel komutlar:

```bash
cryptopredict fetch --symbol BTCUSDT --interval 1h --start 2026-01-01 --end 2026-10-01 --cache-dir data --json
cryptopredict fetch --symbol ETHUSDT --interval 1h --start 2026-01-01 --end 2026-10-01 --cache-dir data --json

cryptopredict backtest --symbol BTCUSDT --interval 1h --start 2026-01-01 --end 2026-10-01 --cache-dir data --model all --splits 5 --fee-bps 10 --json
cryptopredict backtest --symbol ETHUSDT --interval 1h --start 2026-01-01 --end 2026-10-01 --cache-dir data --model all --splits 5 --fee-bps 10 --json

cryptopredict train --symbol BTCUSDT --interval 1h --start 2026-01-01 --end 2026-10-01 --cache-dir data --model ridge --horizon 1 --out models/btc_ridge.joblib --json
cryptopredict predict --model-path models/btc_ridge.joblib --csv data/BTCUSDT_1h.csv --interval 1h --json
```

Ham CSV'ler (`data/`) ve model dosyası (`*.joblib`) yalnız yerelde tutuldu.
İkisi de `.gitignore` kapsamında olduğu için repoya eklenmedi.

## Walk-forward tahmin metrikleri

MAE ve RMSE log-getiri birimindedir. `relative_mae`, model MAE'sinin her zaman
sıfır getiri öngören `zero` baseline MAE'sine oranıdır; 1'in altı baseline'dan
daha iyidir. Yön isabeti sıfır olmayan tahminin işaretini gerçek getirinin
işaretiyle karşılaştırır. `zero` modelinin %0 yön değeri, tahminlerinin işaretsiz
olmasından kaynaklanır ve diğer modellerle bir sınıflandırma skoru gibi
karşılaştırılmamalıdır.

### BTCUSDT

| Model | MAE | RMSE | relative_mae | Yön isabeti |
|---|---:|---:|---:|---:|
| zero | 0,002855 | 0,004380 | 1,0000 | %0,00 |
| mean | 0,002859 | 0,004383 | 1,0013 | %49,82 |
| ridge | 0,002886 | 0,004398 | 1,0109 | %49,43 |
| ma | 0,002960 | 0,004461 | 1,0367 | %49,23 |
| gbm | 0,003036 | 0,004593 | 1,0634 | %49,87 |
| last | 0,004223 | 0,006212 | 1,4789 | %47,44 |

### ETHUSDT

| Model | MAE | RMSE | relative_mae | Yön isabeti |
|---|---:|---:|---:|---:|
| zero | 0,003640 | 0,005743 | 1,0000 | %0,00 |
| mean | 0,003644 | 0,005749 | 1,0011 | %49,98 |
| ridge | 0,003686 | 0,005782 | 1,0128 | %48,67 |
| ma | 0,003768 | 0,005854 | 1,0352 | %48,54 |
| gbm | 0,003857 | 0,006032 | 1,0598 | %48,67 |
| last | 0,005436 | 0,008079 | 1,4937 | %47,16 |

Her iki varlıkta da en düşük hata `zero` baseline'a aittir. Öğrenilmiş Ridge ve
GBM modelleri dahil hiçbir aktif model `relative_mae < 1` elde edemedi. Yaklaşık
%50 yön isabeti de bir sonraki saatlik getiride güçlü ve kararlı bir yön sinyali
bulunmadığını gösteriyor.

## Backtest sonuçları (10 bps ücret)

Getiriler ve maksimum düşüş (MDD) yüzde olarak verilmiştir. `İşlem`, pozisyon
değişimi sayısıdır; tam alış-satış turu sayısı değildir. Buy-and-hold karşılaştırması
başlangıçta tek giriş ücreti öder. `—` değeri, hiç pozisyon alınmadığı ve getiri
standart sapması sıfır olduğu için Sharpe'ın tanımsız olduğunu gösterir.

### BTCUSDT

| Model | Toplam getiri | Buy-and-hold | Sharpe | B&H Sharpe | MDD | B&H MDD | İşlem |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | %0,00 | %22,94 | — | 1,017 | %0,00 | %29,36 | 0 |
| mean | %0,00 | %22,94 | — | 1,017 | %0,00 | %29,36 | 0 |
| ridge | %-67,75 | %22,94 | -7,093 | 1,017 | %68,33 | %29,36 | 1.273 |
| ma | %-20,26 | %22,94 | -1,125 | 1,017 | %32,01 | %29,36 | 526 |
| gbm | %-85,41 | %22,94 | -10,375 | 1,017 | %85,52 | %29,36 | 2.023 |
| last | %-93,77 | %22,94 | -15,170 | 1,017 | %93,85 | %29,36 | 2.854 |

### ETHUSDT

| Model | Toplam getiri | Buy-and-hold | Sharpe | B&H Sharpe | MDD | B&H MDD | İşlem |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | %0,00 | %36,23 | — | 1,195 | %0,00 | %37,89 | 0 |
| mean | %0,00 | %36,23 | — | 1,195 | %0,00 | %37,89 | 0 |
| ridge | %-76,64 | %36,23 | -6,620 | 1,195 | %76,84 | %37,89 | 1.178 |
| ma | %-24,01 | %36,23 | -0,950 | 1,195 | %38,43 | %37,89 | 535 |
| gbm | %-93,51 | %36,23 | -12,336 | 1,195 | %93,55 | %37,89 | 2.208 |
| last | %-93,23 | %36,23 | -10,694 | 1,195 | %93,33 | %37,89 | 2.866 |

`mean` modelinin her iki backtestte de %0 getiri ve sıfır işlem üretmesi, bu
katlardaki ortalama tahminlerin long/flat kuralında pozisyon açtırmamasından
kaynaklanır. Bu, modelin yön metriğinin yaklaşık %50 olmasına rağmen işlem
yapmamasının nedenidir.

## Ücret duyarlılığı

Aynı out-of-sample tahminler 0 bps ücretle yeniden backtest edildi. Aşağıdaki
tablo yalnız işlem yapan modellerin toplam getirisini gösterir:

| Sembol | Model | 0 bps | 10 bps | İşlem |
|---|---|---:|---:|---:|
| BTCUSDT | ridge | %15,24 | %-67,75 | 1.273 |
| BTCUSDT | ma | %34,94 | %-20,26 | 526 |
| BTCUSDT | gbm | %10,38 | %-85,41 | 2.023 |
| BTCUSDT | last | %8,35 | %-93,77 | 2.854 |
| ETHUSDT | ridge | %-24,07 | %-76,64 | 1.178 |
| ETHUSDT | ma | %29,77 | %-24,01 | 535 |
| ETHUSDT | gbm | %-40,84 | %-93,51 | 2.208 |
| ETHUSDT | last | %19,13 | %-93,23 | 2.866 |

Sonuç, saatlik sinyalin ekonomik değerinin işlem maliyetine son derece hassas
olduğunu gösteriyor. Özellikle 1.000-2.800 arası pozisyon değişimi, ücret öncesi
pozitif görünen sonuçları dahi negatife çeviriyor. Ücretsiz senaryo uygulanabilir
bir piyasa varsayımı değildir; ayrıca spread ve slippage burada sıfır kabul
edildiğinden 10 bps sonuçları bile iyimser olabilir.

## Train ve predict örneği

BTCUSDT Ridge modeli 6.519 tam özellik/hedef satırında eğitildi. Eğitim aralığı
2026-01-02 09:00 ile 2026-09-30 23:00 UTC idi. Kaydedilen model, CSV'deki son
bar için aşağıdaki çıktıyı üretti:

| Alan | Değer |
|---|---:|
| Tahmin zamanı (`as_of`) | 2026-10-01 00:00 UTC |
| Son kapanış | 83.509,03 USDT |
| Tahmini log-getiri | -0,000017876 |
| Tahmini yüzde değişim | %-0,001788 |
| Beklenen fiyat | 83.507,54 USDT |
| Yön | aşağı |
| Hedef zamanı | 2026-10-01 02:00 UTC |

Tahminin büyüklüğü pratikte sıfıra çok yakındır; bu tek gözlem bir başarı ölçütü
değildir ve model karşılaştırması yerine kullanılamaz.

## Karşılaşılan durumlar ve iyileştirme önerileri

Tüm istenen `fetch`, `backtest`, `train` ve `predict` komutları hatasız tamamlandı.
Ana Binance uç noktası erişilebildi; alternatif `data-api.binance.vision` veya
`api.binance.us` taban adreslerine ihtiyaç olmadı. İki seride de bu aralık için
boş mum uyarısı görülmedi.

Sonuçların dürüst yorumu, kısa vadeli kripto getirilerinin bu özellikler ve
klasik modeller açısından büyük ölçüde gürültü olduğudur. Sonraki çalışmalar:

1. İşlem sayısını azaltan nötr sinyal eşiğini yalnız eğitim/validasyon
   pencerelerinde seçmek; test dönemine göre eşik ayarlamamak.
2. Ücret, spread ve slippage'ı eğitim/model seçimi hedefinin parçası yapmak;
   günlük işlem limiti veya minimum elde tutma süresi denemek.
3. Ham fiyat birimindeki MACD'yi fiyata bölerek normalize etmek ve farklı fiyat
   seviyeleri/varlıklar arasında daha karşılaştırılabilir kılmak.
4. Hiperparametreleri iç içe zaman serisi doğrulamasıyla seçmek; daha uzun ve
   farklı piyasa rejimlerini kapsayan tamamen ayrılmış son test dönemi tutmak.
5. Yön metriğinde sıfır tahminleri ayrı raporlamak ve olasılık kalibrasyonu,
   precision/recall ile ekonomik eşik metriklerini eklemek.
6. Daha uzun tahmin ufukları ve daha düşük frekansları denemek; saatlik yüksek
   devir hızının maliyet dezavantajını karşılaştırmak.

**Sonuç: Bu rapor ve yazılım yatırım tavsiyesi değildir.**
