# Dashboard ve Docker dağıtımı

Bu rehber, CryptoPredict yönetim panelini GitHub Actions ile imaj olarak
üretmek ve Docker Compose ile çalıştırmak içindir. Dashboard veri çekme, eğitim,
tahmin ve backtest işlemlerini aynı serviste toplar.

## 1. GitHub workflow'larını etkinleştirme

Workflow şablonları, mevcut otomasyon token'ında workflow yazma izni olmadığı
için repoda `deploy/github-workflows/` altında tutulur. GitHub Actions'ın bunları
çalıştırması için iki dosyanın varsayılan dalda şu konumlarda bulunması gerekir:

```text
deploy/github-workflows/ci.yml      -> .github/workflows/ci.yml
deploy/github-workflows/docker.yml  -> .github/workflows/docker.yml
```

Bunu GitHub web arayüzünde yapabilir veya workflow dosyası yazabilen bir
credential kullanabilirsiniz. Fine-grained PAT kullanılıyorsa ilgili repo için
`Workflows: Read and write`; PR'ı otomatik açmak isteniyorsa ayrıca
`Pull requests: Read and write` izni gerekir.

`docker.yml` şu davranışa sahiptir:

- `main` veya `v*` tag push'unda çoklu-platform imajı registry'lere gönderir ve
  `production` environment'ını kullanır.
- Pull request ve elle `workflow_dispatch` çalıştırmalarında imajı yalnızca
  build eder; registry'ye göndermez.
- GHCR için GitHub'ın yerleşik `GITHUB_TOKEN` değerini kullanır. Ayrı bir GHCR
  secret'ı oluşturulmaz.

### `production` environment'ı

GitHub'da **Settings → Environments → New environment** yolunu açın ve adı tam
olarak `production` olan bir environment oluşturun. Environment protection
rule'ları isteğe bağlıdır; zorunlu reviewer eklenirse her imaj yayını onay
bekler.

**Environment variables** bölümündeki değerlerin tamamı opsiyoneldir:

| Ad | Zorunluluk | Örnek / varsayılan | Açıklama |
|---|---|---|---|
| `IMAGE_NAME` | Opsiyonel | `aydinozturk/cyrpto-predict` | GHCR owner/image yolu. Boşsa GitHub repo adı kullanılır; `ghcr.io/` öneki de kabul edilir ve ad küçük harfe çevrilir. |
| `PLATFORMS` | Opsiyonel | `linux/amd64,linux/arm64` | Buildx hedef platformları. Boşsa örnekteki iki platform kullanılır. |
| `DOCKERHUB_USERNAME` | Docker Hub için gerekli | `kullaniciadi` | Doluysa ve `DOCKERHUB_TOKEN` da varsa aynı imaj Docker Hub'a da gönderilir. |
| `DOCKERHUB_REPO` | Opsiyonel | `cyrpto-predict` | Docker Hub repository adı. |

Bu environment değerleri yalnız `production` kullanan publish job'ına uygulanır.
PR ve `workflow_dispatch` build'lerinde aynı override'lar gerekiyorsa bunları
ayrıca repo düzeyinde **Settings → Secrets and variables → Actions → Variables**
altına ekleyin; aksi halde workflow varsayılanları kullanılır.

**Environment secrets**:

| Ad | Zorunluluk | Örnek | Açıklama |
|---|---|---|---|
| `DOCKERHUB_TOKEN` | Docker Hub için gerekli | Docker Hub access token | Yalnız `DOCKERHUB_USERNAME` ile birlikte kullanılır. GHCR-only kurulumda eklemeyin. |

`GITHUB_TOKEN` GitHub tarafından her run için otomatik sağlanır; Settings ekranına
secret olarak eklenmez. Workflow'un `packages: write` izni GHCR'a yazmak için
yeterlidir.

Yayınlanan etiketler varsayılan dal için `latest`, her build için
`sha-<kısa-sha>` ve `v1.2.3` gibi tag'lerde `1.2.3` ile `1.2` biçimleridir.

## 2. Runtime `.env` ayarları

Örnek dosyayı kopyalayın ve yalnız gereken değerleri değiştirin:

```bash
cp .env.example .env
```

`.env` hassas değer içerebilir; repoya commit etmeyin. `.env.example` ile
tanımlanan adların tamamı aşağıdadır:

| Ad | Zorunluluk | Varsayılan | Açıklama |
|---|---|---|---|
| `CRYPTOPREDICT_IMAGE` | Opsiyonel | `ghcr.io/aydinozturk/cyrpto-predict:latest` | Compose'un çalıştıracağı imaj. Tekrarlanabilir dağıtım için `:0.1.0` veya `:sha-...` ile sabitleyin. |
| `CRYPTOPREDICT_BIND` | Opsiyonel | `127.0.0.1` | Host dinleme adresi. `0.0.0.0` ağdan erişim açar; token olmadan kullanmayın. |
| `CRYPTOPREDICT_PUBLISH_PORT` | Opsiyonel | `8000` | Host portu; container portu her zaman `8000`'dir. |
| `CRYPTOPREDICT_DASHBOARD_TOKEN` | Opsiyonel, ağ erişiminde kuvvetle önerilir | boş | `/api/health` dışındaki `/api/*` çağrıları için Bearer token. |
| `CRYPTOPREDICT_DEFAULT_SYMBOL` | Opsiyonel | `BTCUSDT` | Formlarda başlangıçta seçili sembol. |
| `CRYPTOPREDICT_DEFAULT_INTERVAL` | Opsiyonel | `1h` | Formlarda başlangıçta seçili mum aralığı. |
| `CRYPTOPREDICT_BINANCE_URL` | Opsiyonel | `https://api.binance.com` | Binance REST adresi veya erişilebilir bölgesel ayna. |
| `CRYPTOPREDICT_DATA_DIR` | Compose'ta değiştirmeyin | `/data/cache` | Container veri dizini; compose ve imaj tarafından sabitlenir. Docker dışı çalıştırmada varsayılan `./data`dır. |
| `CRYPTOPREDICT_MODEL_DIR` | Compose'ta değiştirmeyin | `/data/models` | Container model dizini; compose ve imaj tarafından sabitlenir. Docker dışı çalıştırmada varsayılan `./models`dır. |
| `CRYPTOPREDICT_HOST` | Compose'ta değiştirmeyin | `0.0.0.0` | Container dinleme adresi; compose tarafından sabitlenir. |
| `CRYPTOPREDICT_PORT` | Compose'ta değiştirmeyin | `8000` | Container portu; compose tarafından sabitlenir. Host portu için `CRYPTOPREDICT_PUBLISH_PORT` kullanın. |

Güçlü bir dashboard token'ı üretmek için:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Çıktıyı `.env` içindeki `CRYPTOPREDICT_DASHBOARD_TOKEN=` satırına yazın.
Dashboard giriş formu token'ı tarayıcının local storage alanında saklar.

## 3. Docker Compose ile çalıştırma

Docker Engine ve Docker Compose **2.24 veya üstü** gerekir; compose dosyası
opsiyonel `.env` için `env_file.required` kullanır.

### GHCR imajını çalıştırma

Workflow imajı yayınladıktan sonra:

```bash
docker compose pull dashboard
docker compose up -d --no-build dashboard
docker compose ps
curl --fail http://127.0.0.1:8000/api/health
```

`CRYPTOPREDICT_PUBLISH_PORT` değerini değiştirdiyseniz URL'de aynı portu
kullanın. Ardından tarayıcıda `http://127.0.0.1:8000` adresini açın.

GHCR paketi private ise önce read-only paket erişimli bir GitHub PAT oluşturun
(`read:packages`) ve komutun parola istemine PAT'i girin:

```bash
docker login ghcr.io -u GITHUB_KULLANICI_ADI
```

Paketin görünürlüğünü GitHub'daki ilgili container package sayfasının package
settings bölümünden kontrol edin. Public pakette pull için login gerekmez;
private pakette kullanıcıya hem repository hem package okuma erişimi verin.

### Yerel checkout'tan build

Registry imajı yerine mevcut kaynak kodunu kullanmak için:

```bash
docker compose up -d --build dashboard
docker compose ps
curl --fail http://127.0.0.1:8000/api/health
```

İlk build, Python/scikit-learn bağımlılıkları nedeniyle birkaç dakika sürebilir.

### Log, güncelleme ve volume

```bash
# Canlı loglar
docker compose logs -f dashboard

# Registry'deki yeni imaja güncelle
docker compose pull dashboard
docker compose up -d --no-build dashboard

# Container'ı durdur; veri/model volume'ünü korur
docker compose down
```

OHLCV önbelleği ve eğitilmiş modeller named volume içindeki `/data/cache` ve
`/data/models` dizinlerinde kalır. `docker compose down -v` bu volume'ü ve
içindeki verileri siler; veri kaybını kabul etmeden bu komutu kullanmayın.

## 4. Dashboard'dan uçtan uca test

1. Paneli açın. Token tanımladıysanız üstteki **Erişim anahtarı** alanına aynı
   değeri yazıp **Kaydet** düğmesine basın.
2. **Veri** sekmesinde `BTCUSDT`, `1h` ve uygun bir başlangıç tarihi seçip
   **Çek** düğmesine basın. **İşler** sekmesinde iş `done` olana kadar bekleyin;
   tamamlanınca mum grafiği oluşur.
3. **Eğit** sekmesinde aynı sembol/aralığı, `ridge` modelini ve ufuk `1`i seçin.
   İsterseniz yalnız harf, rakam, `_`, `.` ve `-` içeren bir model adı verin.
   Eğitim işinin `done` olduğunu doğrulayın.
4. **Modeller** sekmesinde yeni modeli bulun ve **Tahmin et** düğmesine basın.
   Son kapanış, beklenen fiyat, tahmini değişim ve yön kartının geldiğini kontrol
   edin.
5. **Backtest** sekmesinde aynı veriyle önce `ridge`, sonra isterseniz `all`
   seçerek çalıştırın. İş tamamlanınca model metrikleri ile strateji ve
   buy-and-hold sonuçlarını karşılaştırın.

Bu işlemler yatırım tavsiyesi üretmez; panel araştırma ve test amaçlıdır.

## 5. Sınırlamalar ve sorun giderme

- Fetch, train ve backtest işleri tek process içindeki tek worker kuyruğunda
  çalışır. Container yeniden başlarsa kuyruk ve iş geçmişi kaybolur; veri ve
  model dosyaları volume'de kalır. Birden fazla Uvicorn worker çalıştırmayın.
- `CRYPTOPREDICT_DASHBOARD_TOKEN` boşsa health dışındaki API uçları da
  korumasızdır. Bu durumda `CRYPTOPREDICT_BIND=127.0.0.1` değerini değiştirmeyin.
- Binance HTTP 451 döndürürse `.env` içinde
  `CRYPTOPREDICT_BINANCE_URL=https://data-api.binance.vision` gibi erişilebilir
  bir ayna ayarlayın ve container'ı yeniden oluşturun.
- `docker compose` `env_file.required` alanını tanımıyorsa Compose'u 2.24 veya
  daha yeni bir sürüme yükseltin.
- `pull access denied` hatasında imaj adı/tag'ini ve GHCR package görünürlüğünü
  kontrol edin; private paket için yukarıdaki `docker login ghcr.io` adımını
  uygulayın.

## 6. Bu ortamda doğrulananlar

- Workflow dosyaları `actionlint 1.7.12` ve YAML parse kontrolünden geçti;
  tam dashboard entegrasyonunda Python testleri `132 passed` sonucu verdi.
  Workflow'lar `.github/workflows/`
  altına taşınmadığı için GitHub Actions run'ı henüz çalıştırılmadı.
- `docker compose config`, Docker dosyaları birleştirildikten sonra `main`
  üzerinde Compose 2.29.7 ile başarılı oldu. Wheel build, web extra kurulumu,
  health/dashboard yanıtları ve fixture verisiyle train → predict akışı venv
  içinde container adımları taklit edilerek doğrulandı.
- Bu çalışma ortamı mount namespace izni vermediği için gerçek
  `docker compose build`, `pull` ve `up` komutları burada çalıştırılamadı; bu
  bölümdeki gerçek container komutlarının hedef Docker host üzerinde denenmesi
  gerekir.
