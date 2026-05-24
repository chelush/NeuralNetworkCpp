# NeuralNetwork

Библиотека полносвязных нейросетей на C++ с Eigen (курсовой проект).

## Сборка и запуск

```bash
cmake -B build -S .
cmake --build build
./build/demo [epochs] [batch_size] [train_limit] [test_limit] [lr]
```

По умолчанию `demo` читает:

- `data/mnist_train.csv`
- `data/mnist_test.csv`

Скачать CSV можно отсюда:

- [mnist_train.csv](https://pjreddie.com/media/files/mnist_train.csv)
- [mnist_test.csv](https://pjreddie.com/media/files/mnist_test.csv)

Формат строк: `label,pixel1,...,pixel784`.

Пример запуска:

```bash
./build/demo 5 64 60000 10000 0.001
```

Ожидаемый вывод:

- `Epoch N/M, loss = ..., test AUC-ROC = ...`
- `Final test AUC-ROC: ...`
