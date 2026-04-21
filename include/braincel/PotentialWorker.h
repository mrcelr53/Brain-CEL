/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#ifndef POTENTIALWORKER_H
#define POTENTIALWORKER_H

#include "build.h"
#include "utils.h"

#include <QMutex>
#include <random>

enum InputType {
    ConstantScalar,
    RandomScalar,
    CustomScalar,
    PoissonSpikes,
    CustomSpikes,
    Unknown
};
inline InputType toInputType(const QString& str) {
    if (str == "Constant Scalar") { return ConstantScalar; }
    if (str == "Random Scalar") { return RandomScalar; }
    if (str == "Custom Scalar") { return CustomScalar; }
    if (str == "Poisson Spikes") { return PoissonSpikes; }
    if (str == "Custom Spikes") { return CustomSpikes; }
    return Unknown;
}
inline QString toString(const InputType inputType) {
    switch (inputType) {
        case ConstantScalar: return "Constant Scalar";
        case RandomScalar: return "Random Scalar";
        case CustomScalar: return "Custom Scalar";
        case PoissonSpikes: return "Poisson Spikes";
        case CustomSpikes: return "Custom Spikes";
        default: return "Unknown";
    }
};

class PotentialWorker final : public QObject {
    Q_OBJECT

public:
    explicit PotentialWorker();

    void setNodeParams(const QJsonObject &params);
    void setBuildParams(const QJsonObject &params) { buildParams = params; }
    void setSimulationParams(const QJsonObject &params) { simParams = params; }
    QVector<double> potentialTrace() const { return convertToQVector(potential); }
    QVector<double> currentTrace() const { return convertToQVector(current); }
    QVector<double> timeTrace() const { return convertToQVector(time); }

    Network* network() const { return net; }

    InputType inputType() const { return inputType_; }
    QString inputTypeStr() const { return toString(inputType_); }
    void setInputType(const QString& type) { inputType_ = toInputType(type); }
    void setInputType(const InputType& type) { inputType_ = type; }

    void setForceCPUCompute(bool force) { forceCpu_ = force; }
    void setForceCUDACompute(bool force) { forceCuda_ = force; }

signals:
    void potentialResultUpdated(const QVector<double>& time, const QVector<double>& potential, const QVector<double>& current);
    void statusUpdate(const QString& message);
    void statusError(const QString& message);

public slots:
    void build();
    void simulate();
    void reset();

private:
    Network *net;
    QJsonObject simParams = QJsonObject();
    QJsonObject buildParams = QJsonObject();

    bool buildComplete_ = false;
    bool useCudaMembrane_ = false;
    bool forceCpu_ = false;
    bool forceCuda_ = false;

    InputType inputType_ = ConstantScalar;
    Timed* neuron_ = nullptr;
    std::vector<double> time;
    std::vector<double> potential;
    std::vector<double> current;
};


#endif //STDPWORKER_H
