# Laporan Proyek Machine Learning - Sistem Rekomendasi Film

## Project Overview

**Latar Belakang**

Industri hiburan digital, khususnya layanan streaming seperti Netflix, Disney+, dan Amazon Prime, mengalami pertumbuhan eksponensial dalam dekade terakhir. Laporan oleh Statista (2023) menyebutkan bahwa pendapatan global dari pasar layanan video-on-demand diproyeksikan mencapai lebih dari \$150 miliar pada tahun 2025. Salah satu tantangan utama dalam platform ini adalah bagaimana membantu pengguna menemukan film atau serial yang sesuai dengan preferensi mereka dari ribuan judul yang tersedia.

Pengalaman pengguna dapat sangat dipengaruhi oleh kualitas sistem rekomendasi. Tanpa rekomendasi yang relevan, pengguna dapat merasa kewalahan, kehilangan minat, atau berpindah ke platform lain. Oleh karena itu, sistem rekomendasi tidak hanya menjadi fitur tambahan, tetapi merupakan elemen inti dalam meningkatkan retensi dan kepuasan pengguna.

**Permasalahan**

Sistem rekomendasi berbasis popularitas atau manual curation terbukti tidak cukup adaptif terhadap preferensi unik masing-masing pengguna. Selain itu, pendekatan seperti content-based filtering menghadapi keterbatasan pada konteks cold-start (produk/film atau pengguna baru) serta kurang mampu menangkap pola kompleks antar pengguna.

**Solusi yang Diusulkan**

Untuk mengatasi permasalahan tersebut, proyek ini mengembangkan sistem rekomendasi menggunakan dua pendekatan utama: **Content-Based Filtering** dan **Collaborative Filtering berbasis Deep Learning**. Content-based filtering memanfaatkan fitur film untuk menyarankan item serupa, sedangkan collaborative filtering dengan neural network mempelajari pola preferensi pengguna berdasarkan riwayat rating.

Beberapa studi mendukung efektivitas pendekatan ini:

* He et al. (2017) dalam penelitiannya "Neural Collaborative Filtering" menunjukkan bahwa pendekatan deep learning mampu meningkatkan akurasi prediksi dibandingkan metode matrix factorization tradisional.
* Zhang et al. (2019) dalam survei "Deep Learning-based Recommender Systems" menegaskan bahwa embedding dalam neural network dapat menangkap relasi non-linear yang lebih dalam antar entitas pengguna dan item.

**Dampak yang Diharapkan**

* Meningkatkan akurasi rekomendasi berdasarkan interaksi pengguna historis.
* Menangani cold-start problem lebih baik melalui generalisasi embedding.
* Meningkatkan durasi keterlibatan pengguna, click-through rate, dan loyalitas pengguna.
* Menjadi fondasi untuk pengembangan sistem hybrid di masa depan (menggabungkan CF dan content-based).

Referensi:

* He, Xiangnan, et al. (2017). "Neural Collaborative Filtering." Proceedings of the 26th International Conference on World Wide Web (WWW).
* Zhang, Shuai, et al. (2019). "Deep Learning based Recommender System: A Survey and New Perspectives." ACM Computing Surveys (CSUR), 52(1), 1-38.
* Statista Research Department. (2023). "Video-on-Demand Worldwide – Market Overview."

---

## Business Understanding

### Problem Statements

1. Bagaimana merekomendasikan film yang relevan kepada pengguna berdasarkan interaksi sebelumnya?
2. Bagaimana sistem menangani banyaknya film yang belum pernah ditonton oleh pengguna?
3. Dapatkah sistem belajar preferensi pengguna dengan pendekatan pembelajaran mendalam?

### Goals

* Membangun sistem rekomendasi berbasis **deep collaborative filtering**.
* Meningkatkan personalisasi rekomendasi menggunakan embedding pengguna dan item.
* Memberikan hasil rekomendasi yang relevan dengan preferensi historis pengguna.

### Solution Approach

Dalam proyek ini, dua pendekatan sistem rekomendasi dieksplorasi sebagai upaya untuk mencapai tujuan meningkatkan akurasi dan relevansi rekomendasi bagi pengguna, yaitu:

#### 1. Content-Based Filtering (CBF)

Pendekatan content-based filtering berfokus pada pencocokan karakteristik item (dalam hal ini film) dengan preferensi pengguna berdasarkan interaksi sebelumnya. Dalam pendekatan ini, sistem akan merekomendasikan film yang memiliki fitur serupa dengan film yang pernah disukai oleh pengguna.

**Langkah-langkah:**

* Menyusun profil pengguna berdasarkan riwayat film yang diberi rating tinggi.
* Menganalisis fitur film seperti genre, aktor, atau deskripsi (bisa diperluas dengan data tambahan).
* Menghitung kemiripan antar film menggunakan teknik seperti cosine similarity atau TF-IDF (jika tersedia data teks).
* Memberikan rekomendasi berdasarkan film dengan fitur yang paling mirip dengan preferensi pengguna.

**Kelebihan:**

* Tidak membutuhkan data dari pengguna lain (dapat bekerja secara independen).
* Cocok untuk cold-start pengguna baru yang telah berinteraksi sedikit.

**Keterbatasan:**

* Kualitas rekomendasi sangat tergantung pada fitur konten yang tersedia.
* Rentan terhadap overspecialization (merekomendasikan film yang terlalu mirip).

#### 2. Collaborative Filtering (CF) dengan Deep Learning

Pendekatan collaborative filtering berfokus pada hubungan antar pengguna dan item. Dengan menggunakan metode pembelajaran mendalam (deep learning), pendekatan ini mempelajari representasi vektor (embedding) dari pengguna dan item dalam ruang laten.

**Langkah-langkah:**

* Melakukan encoding terhadap `user_id` dan `movie_id` ke indeks integer.
* Membuat model embedding untuk pengguna dan film dengan dimensi tertentu (misalnya 50).
* Menggabungkan embedding melalui dot product dan menambahkan bias term.
* Menggunakan fungsi aktivasi sigmoid untuk memetakan prediksi ke rentang \[0,1].
* Melatih model dengan data interaksi untuk meminimalkan error (menggunakan loss function seperti Binary Crossentropy).

**Kelebihan:**

* Dapat menangkap hubungan kompleks antara pengguna dan item.
* Mampu mempelajari pola preferensi pengguna secara mendalam dari data interaksi historis.

**Keterbatasan:**

* Memerlukan cukup banyak data interaksi untuk melatih model.
* Kurang optimal untuk pengguna atau item yang benar-benar baru (cold-start).

### Ringkasan:

Pemilihan akhir menggunakan collaborative filtering berbasis deep learning karena ketersediaan data interaksi rating yang cukup serta keunggulan model dalam belajar representasi laten yang kompleks. Namun, pendekatan content-based dapat menjadi pelengkap dalam sistem hybrid di masa depan.

## Data Understanding

### Sumber Dataset

Dataset diperoleh dari Kaggle : [Movie recommender system](https://www.kaggle.com/datasets/dev0914sharma/dataset)

Menggunakan dua sumber dataset sebagai berikut :

* **Movie\_Id\_Titles.csv**: Berisi daftar judul film dan ID-nya.
* **Dataset.csv**: Berisi data interaksi pengguna dengan film, termasuk `user_id`, `item_id`, `rating`, dan `timestamp`.

### Deskripsi Kolom:

| Kolom Tipe Deskripsi |         |                                             |
| -------------------- | ------- | ------------------------------------------- |
| user\_id             | Integer | ID unik pengguna                            |
| item\_id             | Integer | ID unik film                                |
| rating               | Float   | Nilai rating yang diberikan pengguna (1–5)  |
| timestamp            | Integer | Waktu interaksi dalam format UNIX           |
| title                | String  | Nama film (dari file Movie\_Id\_Titles.csv) |

### Statistik Awal:

* Total interaksi: **49282** 
* Jumlah pengguna unik: **944**
* Jumlah film unik: **1,682**
* Rentang rating: **1.0 – 5.0**

Distribusi rating dan item divisualisasikan untuk memahami skala data dan persebaran interaksi.

---

### Exploratory Data Analysis (EDA)

#### 1. Distribusi Interaksi Film

Distribusi `item_id` menunjukkan film mana yang paling banyak dinilai pengguna.

```
sns.histplot(data=rating_df, x='item_id', bins=30, kde=True)
plt.title('Distribusi Interaksi per Film')
plt.xlabel('ID Film')
plt.ylabel('Jumlah Interaksi')
plt.show()

```
![ss4](https://github.com/user-attachments/assets/2c23cee5-6d86-4dc2-b22c-225f3edc8e06)

**Insight**:

* Beberapa film mendapatkan jumlah interaksi yang jauh lebih tinggi dari yang lain.
* Kemungkinan besar film populer atau film dengan rating tinggi.

#### 2. Distribusi Rating Berdasarkan Pengguna dan Film

Fungsi `plot_rating_distribution()` digunakan untuk menganalisis distribusi rating berdasarkan `user_id` dan `item_id`.

```
plot_rating_distribution(rating_df, title='Distribusi Rating')
plot_rating_distribution(rating_df, groupby_col='user_id', title='Distribusi Rating Berdasarkan User ID')
plot_rating_distribution(rating_df, groupby_col='item_id', title='Distribusi Rating Berdasarkan Film')

```

![ss1](https://github.com/user-attachments/assets/a0267d78-b51b-47b6-b45e-ac1dff12713e)
![ss2](https://github.com/user-attachments/assets/9aa58dde-a970-4a80-8d66-f45dc070ef34)
![ss3](https://github.com/user-attachments/assets/f674d45d-fc09-4dcb-a37a-d0080841d90c)

**Insight**:

* Beberapa pengguna memberikan lebih banyak rating daripada yang lain.
* Distribusi rating per film menunjukkan bahwa hanya sebagian film yang memiliki interaksi tinggi.

#### 3. Pemeriksaan Duplikasi dan Missing Values

```
movie_df.isnull().sum()
movie_df.duplicated().sum()

```

**Insight**:

* Tidak ditemukan missing value setelah merge.
* Tidak ditemukan duplikasi data.

---

## Data Preparation

Proses data preparation merupakan tahap penting sebelum pembangunan model, terutama dalam sistem rekomendasi berbasis deep learning. Tujuannya adalah untuk memastikan data yang digunakan bersih, konsisten, dan dalam format yang dapat diproses oleh model. Berikut adalah tahapan yang dilakukan beserta alasannya:

1. **Merging Data**
   Dataset `rating_df` yang berisi interaksi pengguna dan `title_df` yang berisi informasi judul film digabung menggunakan `item_id` sebagai key.
   **Alasan**: Menggabungkan informasi film ke dalam dataset utama memungkinkan analisis dan interpretasi hasil yang lebih kaya, serta mendukung sistem rekomendasi yang dapat menampilkan judul film saat inferensi.
2. **Pembersihan Data**

   * Menghapus kolom `timestamp` karena tidak digunakan dalam analisis atau model.
   * Mengecek dan memastikan tidak ada nilai null atau duplikat.
     **Alasan**: Menghindari gangguan dalam training dan menjaga kualitas data. Kolom yang tidak relevan dihapus untuk mengurangi kompleksitas.
3. **Encoding ID ke Integer**

   * `user_id` dan `movie_id` diubah ke format indeks integer.
   * Mapping dilakukan menggunakan dictionary agar ID numerik unik dapat digunakan dalam embedding layer.
     **Alasan**: Layer embedding pada model deep learning hanya menerima input dalam bentuk integer indeks. Encoding ini penting untuk proses pembelajaran representasi laten pengguna dan item.
4. **Standarisasi Rating**
   Rating dinormalisasi ke rentang \[0, 1] menggunakan metode min-max scaling.
   **Alasan**: Model menggunakan fungsi aktivasi sigmoid pada output sehingga memerlukan nilai target dalam rentang \[0,1] untuk konsistensi prediksi.
5. **Data Splitting**
   Dataset diacak secara acak (`random_state=42`) dan dibagi menjadi 80% data latih dan 20% data validasi.
   **Alasan**: Memisahkan data pelatihan dan pengujian sangat penting untuk mengevaluasi performa model secara adil dan menghindari overfitting. Pengacakan data dilakukan agar distribusi rating merata di kedua subset.

Dengan melalui proses di atas, data menjadi siap digunakan untuk melatih model rekomendasi berbasis collaborative filtering. Setiap langkah berkontribusi pada peningkatan kualitas data, kestabilan pelatihan model, dan akurasi prediksi yang dihasilkan.

1. **Merging Data**
   Dataset `rating_df` dan `title_df` digabung berdasarkan `item_id` untuk menggabungkan nama film ke data interaksi.
2. **Pembersihan Data**

   * Kolom `timestamp` dihapus karena tidak digunakan.
   * Tidak terdapat missing value atau duplikat setelah penggabungan.
3. **Encoding ID ke Integer**

   * `user_id` dan `movie_id` diubah menjadi indeks integer dengan teknik dictionary mapping.
   * Ini penting untuk keperluan embedding layer dalam deep learning.
4. **Standarisasi Rating**

   * Rating diskalakan ke rentang \[0, 1] menggunakan normalisasi min-max agar sesuai dengan output fungsi aktivasi sigmoid.
5. **Data Splitting**

   * Data diacak dan dibagi menjadi 80% training dan 20% validasi.

---

## Modeling

### Arsitektur Model: Deep Collaborative Filtering

Model dibangun menggunakan TensorFlow dan Keras dengan pendekatan berikut:

* **Input**: pasangan `(user, movie)`
* **Layer Embedding**: merepresentasikan pengguna dan film dalam ruang vektor dimensi 50.
* **Dot Product**: mengukur kedekatan antara pengguna dan film.
* **Bias Term**: untuk menyesuaikan preferensi global.
* **Sigmoid Activation**: untuk menghasilkan output prediksi rating yang diskalakan.

### Kompilasi Model

```
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

```

### Training

Model dilatih selama 10 epoch menggunakan `batch_size=32` dengan hasil RMSE validasi cenderung menurun, menandakan model belajar secara stabil.

---

## Evaluation

Model dievaluasi menggunakan metrik **Root Mean Squared Error (RMSE)**, yang mengukur seberapa jauh prediksi model dari nilai rating sebenarnya.

Hasil pelatihan model selama 10 epoch menunjukkan bahwa nilai RMSE pada data training terus menurun dan cenderung stabil pada data validasi, menandakan proses pembelajaran berlangsung dengan baik.

* RMSE mengukur rata-rata galat prediksi.
* Digunakan baik untuk data training maupun validasi.
  
| Epoch | RMSE (Train) | RMSE (Validation) |
| ----- | ------------ | ----------------- |
| 1     | 0.2881       | 0.2563            |
| 2     | 0.2533       | 0.2473            |
| 3     | 0.2464       | 0.2451            |
| 4     | 0.2438       | 0.2426            |
| 5     | 0.2413       | 0.2423            |
| 6     | 0.2391       | 0.2427            |
| 7     | 0.2413       | 0.2428            |
| 8     | 0.2395       | 0.2425            |
| 9     | 0.2394       | 0.2423            |
| 10    | 0.2396       | 0.2421            |

* RMSE pada data pelatihan menurun secara konsisten, menunjukkan model belajar representasi preferensi pengguna dengan baik.

* RMSE validasi menurun hingga epoch ke-4, lalu mulai stagnan. Ini  menandakan bahwa model mencapai titik stabil dan tidak mengalami overfitting.

### Fungsi Rekomendasi

Model digunakan untuk menghasilkan rekomendasi dengan cara:

1. Mengambil semua film yang belum pernah ditonton oleh pengguna.
2. Memprediksi kemungkinan pengguna menyukai film tersebut menggunakan model.
3. Mengurutkan hasil berdasarkan skor prediksi.
4. Mengambil 10 film teratas sebagai rekomendasi.

Contoh output:

```
Showing recommendations for user: 193
========================================
Movies with high ratings from user
----------------------------------------
- The Matrix - Rating: 5.0
- The Lord of the Rings - Rating: 4.5

Top movie recommendations
----------------------------------------
- Inception
- The Dark Knight
- Interstellar
```

### Kelebihan dan Kelemahan Model

**Kelebihan:**

* Mampu mempelajari pola kompleks antar pengguna dan film melalui embedding.
* Tidak memerlukan informasi konten film (genre, deskripsi, dll).
* Adaptif terhadap pola preferensi baru selama cukup data interaksi tersedia.

**Kekurangan:**

* Tidak optimal untuk pengguna/item baru tanpa interaksi historis (cold-start problem).
* Membutuhkan sumber daya komputasi lebih besar dibanding model berbasis aturan sederhana.

---
---

## Recommendation Generation

Fungsi `get_recommendations(user_id)` digunakan untuk:

1. Mengambil film yang belum ditonton oleh pengguna.
2. Memprediksi rating untuk film tersebut menggunakan model.
3. Menampilkan 10 rekomendasi teratas.

Contoh output:

```
Showing recommendations for user: 193
========================================
Movies with high ratings from user
----------------------------------------
- The Matrix - Rating: 5.0
- The Lord of the Rings - Rating: 4.5

Top movie recommendations
----------------------------------------
- Inception
- The Dark Knight
- Interstellar

```

---

## Conclusion and Future Work

### Kesimpulan

* Sistem rekomendasi berhasil mempelajari preferensi pengguna dan memberikan rekomendasi film secara personal.
* Model embedding mampu menangkap pola kompleks dalam data rating.

### Future Work

* Tambahkan data fitur konten film (genre, tahun, sutradara) untuk hybrid recommender system.
* Lakukan hyperparameter tuning (jumlah embedding, optimizer).
* Implementasi fitur filtering seperti genre-based atau rating threshold.
