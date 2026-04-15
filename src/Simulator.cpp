/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
 */

#include <braincel/Simulator.h>
#include <braincel/SimulationWorker.h>

#include <Python.h>
#include <iostream>
#include <stdexcept>

Simulator::Simulator() {
    Py_Initialize();
    PyEval_SaveThread();
    worker = new SimulationWorker();
}

Simulator::~Simulator() {
    delete worker;
    Py_Finalize();
}
void Simulator::build(const nlohmann::json& params) {
    worker->setBuildParams(params);
    worker->build();
    buildComplete = true;
}

void Simulator::simulate(const nlohmann::json& params) {
    if (!buildComplete)
        throw std::runtime_error("Simulation cannot start before build is complete.");
    worker->setSimulationParams(params);
    worker->simulate();
}

void Simulator::pause() {
    worker->pause();
}

void Simulator::stop() {
    worker->pause();
    worker->reset();
    buildComplete = false;
}

void Simulator::requestConnections(const std::string& preGroup, const std::string& postGroup) {
    auto con = worker->getConnections(preGroup, postGroup);
    // caller can use return value directly — no signal needed
}