# ОПИСАНИЕ АЛГОРИТМА РАБОТЫ ПРОГРАММЫ

## 1. Формализация задачи

Пусть задано множество текстовых сообщений:

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}
$$

где:
- $x_i$ — текстовое сообщение
- $y_i \in \{1, \dots, K\}$ — идентификатор семантического кластера

---

## 2. Построение эмбеддингов

$$
\mathbf{z}_i = f(x_i), \quad \mathbf{z}_i \in \mathbb{R}^d
$$

Множество эмбеддингов:

$$
\mathcal{Z} = \{\mathbf{z}_i\}_{i=1}^{N}
$$

---

## 3. Модель нормы

### Средний вектор

$$
\boldsymbol{\mu}_k =
\frac{1}{N_k}
\sum_{i: y_i = k}
\mathbf{z}_i
$$

### Ковариационная матрица

$$
\Sigma_k =
\frac{1}{N_k - 1}
\sum_{i: y_i = k}
(\mathbf{z}_i - \boldsymbol{\mu}_k)
(\mathbf{z}_i - \boldsymbol{\mu}_k)^T
$$

Регуляризация:

$$
\Sigma_k^{*} = \Sigma_k + \lambda I_d
$$

---

## 4. Расстояние Махаланобиса

$$
d_k(\mathbf{z}) =
\sqrt{
(\mathbf{z} - \boldsymbol{\mu}_k)^T
(\Sigma_k^{*})^{-1}
(\mathbf{z} - \boldsymbol{\mu}_k)
}
$$

---

## 5. Классификация

$$
k^* = \arg\min_k d_k(\mathbf{z})
$$

$$
d_{\min} = d_{k^*}(\mathbf{z})
$$

---

## 6. Порог

$$
\tau_k = Q_q(d_k(\mathbf{z}_i))
$$

---

## 7. Критерий дрейфа

$$
\text{drift} =
\begin{cases}
1, & d_{\min} > \tau_{k^*} \\
0, & \text{иначе}
\end{cases}
$$

$$
\text{score} = d_{\min} - \tau_{k^*}
$$
