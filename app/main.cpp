#include "core/random/random.hpp"
#include "nn/loss/mse.hpp"
#include "nn/model/network_builder.hpp"
#include "nn/optim/adam.hpp"
#include "nn/scheduler/lr_schedule.hpp"
#include "train/train.hpp"
#include "utils/data_loader.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace nn;

namespace {

void print_usage(const char* program) {
    std::cerr << "usage: " << program
              << " [epochs=5] [batch_size=64] [train_limit=60000] [test_limit=10000]"
                 " [lr=0.001]\n";
}

bool is_integer(const std::string& s) {
    if (s.empty())
        return false;
    for (unsigned char c : s) {
        if (!std::isdigit(c))
            return false;
    }
    return true;
}

std::string trim_ascii(std::string value) {
    while (!value.empty() && (value.back() == '\r' || value.back() == '\n' || value.back() == ' ' ||
                              value.back() == '\t'))
        value.pop_back();
    std::size_t begin = 0;
    while (begin < value.size() &&
           (value[begin] == ' ' || value[begin] == '\t' || value[begin] == '\r' ||
            value[begin] == '\n'))
        ++begin;
    return value.substr(begin);
}

double macro_roc_auc(const MatrixXf& scores, const MatrixXf& Y) {
    if (scores.rows() != Y.rows() || scores.cols() != Y.cols())
        throw std::runtime_error("macro_roc_auc(): scores and Y must have the same shape");
    if (scores.rows() == 0)
        throw std::runtime_error("macro_roc_auc(): scores must not be empty");

    double total_auc = 0.0;
    for (Eigen::Index cls = 0; cls < scores.cols(); ++cls) {
        std::vector<std::pair<float, bool>> values;
        values.reserve(static_cast<std::size_t>(scores.rows()));

        Eigen::Index positives = 0;
        for (Eigen::Index row = 0; row < scores.rows(); ++row) {
            const bool positive = Y(row, cls) > 0.5f;
            positives += positive ? 1 : 0;
            values.emplace_back(scores(row, cls), positive);
        }

        const Eigen::Index negatives = scores.rows() - positives;
        if (positives == 0 || negatives == 0) {
            total_auc += 0.5;
            continue;
        }

        std::sort(values.begin(), values.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        double positive_rank_sum = 0.0;
        std::size_t i = 0;
        while (i < values.size()) {
            std::size_t j = i + 1;
            while (j < values.size() && values[j].first == values[i].first)
                ++j;

            const double average_rank = (static_cast<double>(i + 1) + static_cast<double>(j)) / 2.0;
            for (std::size_t k = i; k < j; ++k) {
                if (values[k].second)
                    positive_rank_sum += average_rank;
            }
            i = j;
        }

        total_auc += (positive_rank_sum - static_cast<double>(positives) * (positives + 1) / 2.0) /
                     (static_cast<double>(positives) * negatives);
    }

    return total_auc / scores.cols();
}

double evaluate_auc_roc(Network& net, const MatrixXf& X, const MatrixXf& Y,
                        std::size_t batch_size) {
    if (X.rows() != Y.rows())
        throw std::runtime_error("evaluate_auc_roc(): X and Y must have equal number of samples");
    if (batch_size == 0)
        throw std::runtime_error("evaluate_auc_roc(): batch_size must be > 0");

    MatrixXf scores(X.rows(), Y.cols());
    for (std::size_t start = 0; start < static_cast<std::size_t>(X.rows()); start += batch_size) {
        const std::size_t end = std::min(start + batch_size, static_cast<std::size_t>(X.rows()));
        const Eigen::Index rows = static_cast<Eigen::Index>(end - start);

        MatrixXf batch_X = X.middleRows(static_cast<Eigen::Index>(start), rows);
        scores.middleRows(static_cast<Eigen::Index>(start), rows) = net.forward(std::move(batch_X));
    }

    return macro_roc_auc(scores, Y);
}

std::pair<MatrixXf, MatrixXf> load_mnist_csv(const std::string& path, Eigen::Index limit) {
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("cannot open " + path);

    constexpr Eigen::Index features = 784;
    constexpr Eigen::Index classes = 10;
    constexpr float scale = 1.0f / 255.0f;

    std::vector<float> values;
    std::vector<int> labels;
    labels.reserve(limit > 0 ? static_cast<std::size_t>(limit) : 60000);
    values.reserve(labels.capacity() * features);

    std::string line;
    std::size_t line_no = 0;
    while (std::getline(file, line)) {
        ++line_no;
        if (line.empty())
            continue;
        if (limit > 0 && static_cast<Eigen::Index>(labels.size()) >= limit)
            break;

        std::stringstream row(line);
        std::string token;
        if (!std::getline(row, token, ','))
            continue;
        token = trim_ascii(std::move(token));
        if (!is_integer(token)) {
            if (labels.empty() && line_no == 1)
                continue;  // allow optional CSV header
            throw std::runtime_error("invalid label token at line " + std::to_string(line_no) +
                                     " in " + path + ": '" + token + "'");
        }

        const int label = std::stoi(token);
        if (label < 0 || label >= classes)
            throw std::runtime_error("invalid MNIST label at line " + std::to_string(line_no) +
                                     " in " + path);

        const std::size_t start = values.size();
        while (std::getline(row, token, ',')) {
            const std::string pixel = trim_ascii(std::move(token));
            try {
                values.push_back(std::stof(pixel) * scale);
            } catch (const std::exception&) {
                throw std::runtime_error("invalid pixel token at line " + std::to_string(line_no) +
                                         " in " + path + ": '" + pixel + "'");
            }
        }

        if (values.size() - start != static_cast<std::size_t>(features))
            throw std::runtime_error("expected 784 pixels per row at line " +
                                     std::to_string(line_no) + " in " + path);

        labels.push_back(label);
    }

    MatrixXf X(static_cast<Eigen::Index>(labels.size()), features);
    MatrixXf Y = MatrixXf::Zero(static_cast<Eigen::Index>(labels.size()), classes);
    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        for (Eigen::Index j = 0; j < features; ++j)
            X(i, j) = values[static_cast<std::size_t>(i * features + j)];
        Y(i, labels[static_cast<std::size_t>(i)]) = 1.0f;
    }

    return {std::move(X), std::move(Y)};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 6) {
        print_usage(argv[0]);
        return 1;
    }

    Eigen::Index epochs = 5;
    Eigen::Index batch_size = 64;
    Eigen::Index train_limit = 60000;
    Eigen::Index test_limit = 10000;
    float learning_rate = 0.001f;
    try {
        epochs = argc > 1 ? std::stoll(argv[1]) : epochs;
        batch_size = argc > 2 ? std::stoll(argv[2]) : batch_size;
        train_limit = argc > 3 ? std::stoll(argv[3]) : train_limit;
        test_limit = argc > 4 ? std::stoll(argv[4]) : test_limit;
        learning_rate = argc > 5 ? std::stof(argv[5]) : learning_rate;
    } catch (const std::exception& e) {
        std::cerr << "invalid CLI arguments: " << e.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }
    if (epochs <= 0 || batch_size <= 0 || learning_rate <= 0.0f) {
        std::cerr << "epochs, batch_size and lr must be > 0\n";
        print_usage(argv[0]);
        return 1;
    }

    std::mt19937 engine(42);
    Random rnd(engine);

    const std::string train_path = "data/mnist_train.csv";
    const std::string test_path = "data/mnist_test.csv";

    std::cout << "Loading train dataset...\n";
    auto [train_X, train_Y] = load_mnist_csv(train_path, train_limit);
    std::cout << "Loading test dataset...\n";
    auto [test_X, test_Y] = load_mnist_csv(test_path, test_limit);
    DataLoader train_loader(std::move(train_X), std::move(train_Y),
                            BatchSize{static_cast<std::size_t>(batch_size)}, Shuffle{true});

    Network net = NetworkBuilder::set_input(In{784}, rnd)
                      .add_linear(Out{128})
                      .add_relu()
                      .add_linear(Out{64})
                      .add_relu()
                      .add_linear(Out{10})
                      .extract();

    MSE mse;
    Adam adam;
    CosineAnnealingSchedule schedule(learning_rate, learning_rate * 0.01f, epochs);
    TrainOptions one_epoch{1, 0};

    std::cout << "Start training...\n";
    for (Eigen::Index epoch = 0; epoch < epochs; ++epoch) {
        const float lr = schedule.at(epoch);
        const float loss = train(net, train_loader, mse, adam, lr, one_epoch, engine);
        const double test_auc =
            evaluate_auc_roc(net, test_X, test_Y, static_cast<std::size_t>(batch_size));

        std::cout << "Epoch " << (epoch + 1) << "/" << epochs << ", lr = " << lr
                  << ", loss = " << loss << ", test AUC-ROC = " << test_auc << '\n';
    }

    const double final_test_auc =
        evaluate_auc_roc(net, test_X, test_Y, static_cast<std::size_t>(batch_size));
    std::cout << "Final test AUC-ROC: " << final_test_auc << '\n';

    return 0;
}
