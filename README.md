# RNN & BPTT — From Scratch (NumPy) + PyTorch (BasicRNN/LSTM)

Bu repo, RNN (Recurrent Neural Network) ve **BPTT (Backpropagation Through Time)** kavramlarını iki farklı seviyede pekiştirmek için hazırlanmış bir egzersiz seti içerir:

- **Q1 (`q1_bptt_for_torch/`)**: NumPy ile basit bir RNN hücresi ve lineer katman yazıp _forward_, _BPTT ile backward_, **gradient checking**, **exploding gradients** ve **gradient clipping** mantığını görürsün.
- **Q2 (`q2_rnn_torch/`)**: PyTorch ile kendi `BasicRNN` ve kendi `LSTM` hücreni kullanarak iki oyuncak problem çözer:
  - **Sinüs devam ettirme** (regression)
  - **Memorization / sequence recall** (classification)

> Not: Bu repo eğitim amaçlıdır; kodlar “konuyu öğretmeye” odaklıdır.

## Klasör Yapısı

- `q1_bptt_for_torch/`
  - `BasicRNNCell.py`: NumPy tabanlı basit RNN hücresi (state tutar, BPTT yapar)
  - `LinearLayer.py`: Çıkışa giden lineer katman
  - `common.py`: `softmax`, `cross_entropy_loss`, `clip_gradient`
  - `data_generator.py`: Rastgele veri ve memorization verisi üretimi
  - `train.py`: Eğitim adımı / memorization training döngüsü
  - `test.py`: Gradient check ve exploding gradients görselleştirme
  - `main.py`: Q1 için giriş noktası

- `q2_rnn_torch/`
  - `BasicRNNCell.py`: PyTorch tabanlı BasicRNN hücresi
  - `MyLSTMCell.py`: PyTorch tabanlı LSTM hücresi (kapılar + cell state)
  - `RNNPlusLinearLayer.py`: RNN hücresi + lineer çıkış
  - `SineDataset.py`: Sinüs continuation dataseti
  - `CharSequenceDataset.py`: Memorization dataseti (one-hot)
  - `memory_task.py`: Memorization loss/accuracy
  - `loops.py`: Train/test loop’ları (+ grad clipping)
  - `main.py`: Q2 için giriş noktası

- `rnn_bptt_summary.txt`: Q1/Q2 için kısa teorik özet
- `exercise-6.pdf`, `notes_ex6.pdf`: Egzersiz notları

## Kurulum

Python 3.9+ önerilir.

Gerekli paketler (minimum):

- `numpy`
- `matplotlib`
- `torch`

Kurulum örneği:

```bash
pip install numpy matplotlib torch
```

> CUDA şart değil. `q2_rnn_torch/loops.py` otomatik olarak GPU varsa kullanır.

## Çalıştırma

### Q1 — NumPy ile BPTT / Gradient Check / Clipping

Klasöre girip çalıştırman yeterli:

```bash
python q1_bptt_for_torch/main.py --task 1
```

Q1 görevleri (`--task`):

- `1`: Gradient checking (BPTT türevleri doğru mu?)
- `2`: Exploding gradients görselleştirme (plot)
- `3`: Memorization task eğitimi

Örnek (memorization):

```bash
python q1_bptt_for_torch/main.py --task 3 --nbr_iters 70000 --lr 0.001 --sz_rnn 50 --sz_vocab 10
```

### Q2 — PyTorch ile BasicRNN vs LSTM

Q2 görevleri (`--task`):

- `1`: Sine continuation (MSE loss)
- `2`: Memorization / sequence recall (cross entropy tarzı loss)

RNN tipi (`--rnn_type`): `BasicRNN` veya `LSTM`

Örnekler:

```bash
# Sine continuation
python q2_rnn_torch/main.py --task 1 --rnn_type BasicRNN
python q2_rnn_torch/main.py --task 1 --rnn_type LSTM

# Memorization
python q2_rnn_torch/main.py --task 2 --rnn_type BasicRNN
python q2_rnn_torch/main.py --task 2 --rnn_type LSTM
```

Beklenen çıktı notları:

- Sine testinde `loops.py` grafikleri dosyaya kaydeder: `Out_<RNN_TYPE>_<index>.png`.
- Memorization testinde ortalama test loss ve accuracy yazdırılır.

## GitHub Repo İsmi Önerileri

Kısa ve açıklayıcı birkaç seçenek:

- `rnn-bptt-from-scratch`
- `rnn-bptt-numpy-pytorch`
- `basic-rnn-lstm-exercises`
- `sequence-modeling-rnn-lstm`
- `exercise-6-rnn-bptt`

Repo açıklaması (GitHub “About” kısmı) için öneri:

> “RNN + BPTT öğrenme egzersizleri: NumPy ile BPTT/gradient check/clipping, PyTorch ile BasicRNN ve LSTM üzerinde sine continuation ve memorization task.”

## İpucu

- Q1’de BPTT yazarken en hızlı doğrulama: `--task 1` gradient check.
- Uzun dizilerde gradient patlaması normal; clipping bunun en basit önlemlerinden.
- Memorization task’ta genelde **LSTM**, BasicRNN’e göre daha stabil öğrenir.
