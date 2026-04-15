/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
 */


#ifndef STDPWORKER_H
#define STDPWORKER_H

#include "../../src/build.h"
#include "../utils.h"

#include <QMutex>
#include <random>



class StdpWorker final : public QObject {
    Q_OBJECT

public:
    explicit StdpWorker();

    void setConnectionParams(const QJsonObject &params);
    void setBuildParams(const QJsonObject &params) { buildParams = params; }
    void setSimulationParams(const QJsonObject &params) { simParams = params; }
    QPair<QVector<double>, QVector<double>> stdpWindow() const { return {convertToQVector(dts), convertToQVector(dws)}; }

    Network* network() const { return net; }

signals:
    void stdpResultUpdated(const QVector<double>& dts, const QVector<double>& dws);
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
    double initialWeight = 0.5;

    const int number = 200;
    const float clampPotential = -55.;

    std::vector<double> dts;
    std::vector<double> dws;
};


#endif //STDPWORKER_H
